import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from detection_gligen_cos import prompt_gradient_alignment as alignment
from detection_gligen_cos import retrain_runner as runner
from detection_gligen_cos.prompt_workers import allocate_workers, merge_workers, run_prompt_workers, scaled_steps


class PromptWorkersTest(unittest.TestCase):
    def test_balanced_classes_and_exact_global_budget(self):
        workers = allocate_workers(list(range(80)), ["0", "1", "2", "3"], 1000)
        self.assertEqual([w["steps"] for w in workers], [250] * 4)
        self.assertEqual([len(w["classes"]) for w in workers], [20] * 4)
        self.assertEqual(scaled_steps(100, 250, 1000), 25)
        self.assertEqual(scaled_steps(0, 250, 1000), 0)
        self.assertEqual([w["steps"] for w in allocate_workers(list(range(5)), ["0", "1", "2", "3"], 1000)],
                         [400, 200, 200, 200])
        for count in (1, 2, 5, 79, 80):
            for budget in (1, 3, 1000):
                workers = allocate_workers(list(range(count)), ["0", "1", "2", "3"], budget)
                self.assertEqual(sum(w["steps"] for w in workers), budget)
                self.assertTrue(all(w["steps"] > 0 for w in workers))
                self.assertEqual(sorted(c for w in workers for c in w["classes"]), list(range(count)))

    def write_worker(self, worker, extra=None):
        output = Path(worker["output"])
        output.mkdir(parents=True, exist_ok=True)
        groups = {f"class_{c}": [f"token_{c}"] for c in worker["classes"]}
        embeddings = {f"token_{c}": torch.tensor([float(c)]) for c in worker["classes"]}
        for cls, value in (extra or {}).items():
            groups[f"class_{cls}"] = [f"token_{cls}"]
            embeddings[f"token_{cls}"] = torch.tensor([float(value)])
        name = f"learned_embeds-{worker['steps']}.bin"
        torch.save(embeddings, output / name)
        (output / "object_prompts.json").write_text(json.dumps({
            "groups": groups, "latest": name,
            "train_groups": [f"class_{c}" for c in worker["classes"]],
            "gradient_alignment": {"guide_metadata_hash": "same", "statistics": {"groups": {}}},
        }))

    def test_merge_uses_only_owned_tokens_and_preserves_previous_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workers = [{"classes": [0], "steps": 2, "output": str(root / "w0")},
                       {"classes": [1], "steps": 2, "output": str(root / "w1")}]
            self.write_worker(workers[0], {1: -1, 9: 99})
            self.write_worker(workers[1], {0: -1, 9: 99})
            merge_workers(root, workers, 4, 100)
            learned = torch.load(root / "learned_embeds-4.bin", weights_only=True)
            self.assertEqual({key: value.item() for key, value in learned.items()},
                             {"token_0": 0, "token_1": 1, "token_9": 99})
            metadata = json.loads((root / "object_prompts.json").read_text())
            self.assertEqual(metadata["preserved_resume_groups"], ["class_9"])
            self.assertEqual(metadata["completed_steps"], 4)

    def test_prepare_before_workers_and_resume_completed_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, commands = [], []
            cfg = {"device": "0", "seed": 46, "prompt_optimization": {
                "prompt_scope": "class", "max_train_steps": 1000, "save_steps": 100,
                "lr_warmup_steps": 40, "gradient_alignment": {"warmup_steps": 100}}}
            command = ["python", "-m", "worker", "--max-train-steps", "1000"]

            def prepare(command, *args, **kwargs):
                events.append("prepare")
                self.assertIn("--prepare-workers", command)
                path = Path(command[command.index("--output-dir") + 1])
                path.mkdir(parents=True, exist_ok=True)
                (path / "worker_classes.json").write_text(json.dumps(list(range(8))))

            def launch(command, **kwargs):
                self.assertEqual(events[0], "prepare")
                events.append("worker")
                commands.append(command)
                option = lambda name: command[command.index(name) + 1]
                self.assertEqual(option("--max-train-steps"), "250")
                self.assertEqual(option("--alignment-warmup-steps"), "25")
                self.assertEqual(option("--lr-warmup-steps"), "10")
                self.assertIn("--require-guide-cache", command)
                worker = {"classes": list(map(int, option("--train-class-ids").split(","))),
                          "steps": 250, "output": option("--output-dir")}
                self.write_worker(worker)
                return SimpleNamespace(poll=lambda: 0, wait=lambda: 0)

            with patch.object(runner, "run_command", side_effect=prepare), \
                    patch("detection_gligen_cos.prompt_workers.subprocess.Popen", side_effect=launch):
                run_prompt_workers(command, cfg, root, ["0", "1", "2", "3"], 0)
                run_prompt_workers(command, cfg, root, ["0", "1", "2", "3"], 0)
                self.assertEqual(len(commands), 4)
                self.write_worker({"classes": [0, 4], "steps": 125,
                                   "output": str(root / "_workers/worker_0")})
                run_prompt_workers(command, cfg, root, ["0", "1", "2", "3"], 0)
            self.assertEqual(len(commands), 5)
            self.assertEqual(commands[-1][commands[-1].index("--initial-step") + 1], "125")
            self.assertEqual(commands[-1][commands[-1].index("--resume-mode") + 1], "overwrite")
            self.assertEqual(len(json.loads((root / "object_prompts.json").read_text())["groups"]), 8)

    def test_shared_cache_is_read_only_and_missing_cache_never_computes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = SimpleNamespace(enable_alignment_parameters=lambda: None,
                                       alignment_parameters=lambda: [torch.zeros(2)], device="cpu")
            with patch.object(alignment, "GuideDetectionDataset", return_value=object()), \
                    patch.object(alignment, "build_guide_metadata", return_value={"test": True}), \
                    patch.object(alignment, "compute_guide_gradients") as compute:
                args = (detector, root, root, root, None, [], [], 64, False, 1, 0)
                with self.assertRaisesRegex(RuntimeError, "missing or incompatible"):
                    alignment.load_or_compute_guide_gradients(*args, require_cache=True)
                torch.save({"metadata": {"test": True}, "gradients": [torch.ones(2)]},
                           root / "guide_gradients.pt")
                gradients, metadata = alignment.load_or_compute_guide_gradients(*args, require_cache=True)
                self.assertTrue(metadata["cache_hit"])
                self.assertTrue(torch.equal(gradients[0], torch.ones(2)))
                self.assertFalse((root / "guide_metadata.json").exists())
                compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
