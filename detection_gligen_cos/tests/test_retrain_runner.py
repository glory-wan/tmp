from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from detection_gligen_cos import retrain_runner as runner
from detection_gligen_cos.generate_gligen_sdedit_examples import update_resume_state


class RetrainRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gligen-cos-runner.")
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        for relative in (
            "images/train2017",
            "images/val2017",
            "labels/train2017",
            "labels/val2017",
            "annotation",
        ):
            (self.dataset_root / relative).mkdir(parents=True, exist_ok=True)
        self.dataset_yaml = self.root / "dataset.yaml"
        self.dataset_yaml.write_text(
            f"path: {self.dataset_root}\n"
            "train: images/train2017\n"
            "val: images/val2017\n"
            "names: [umbrella, bird, bus]\n",
            encoding="utf-8",
        )
        self.workspace = self.root / "workspace"

    def tearDown(self):
        self.temporary.cleanup()

    def test_retrain_yaml_uses_all_synthetic_rounds_once(self):
        cfg = {
            "dataset": str(self.dataset_yaml),
            "workspace": str(self.workspace),
        }
        output = runner.retrain_dataset_yaml(cfg, 2)
        payload = yaml.safe_load(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["train"][0], "images/train2017")
        self.assertEqual(
            payload["train"][1:],
            [
                str((self.workspace / "round_1/synthetic/images").resolve()),
                str((self.workspace / "round_2/synthetic/images").resolve()),
            ],
        )
        self.assertNotIn("cos_more_0", output.read_text(encoding="utf-8"))

    def test_round_state_has_no_post_generation_alignment_or_groups(self):
        state = {"rounds": {}}
        record = runner.round_state(state, 1)
        self.assertNotIn("align", record)
        self.assertNotIn("groups", record["retrain"])
        self.assertEqual(record["retrain"]["epoch"], 0)

    def test_schema_one_state_is_rejected(self):
        path = self.root / "old-state.json"
        path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected 2"):
            runner.read_json(path)

    def test_generator_updates_image_level_state(self):
        state_path = self.root / "state.json"
        state = {
            "schema_version": 2,
            "current": {"round": 3, "stage": "generate"},
            "rounds": {"3": {"generate": {"status": "running"}}},
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
        update_resume_state(state_path, generated=1021, target=2000, complete=False)
        updated = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["current"]["generated_samples"], 1021)
        self.assertEqual(
            updated["rounds"]["3"]["generate"]["generation_target"], 2000
        )
        self.assertFalse(updated["current"]["generate_complete"])

    def test_run_generate_invokes_cos_module(self):
        cfg = {
            "workspace": str(self.workspace),
            "dataset": str(self.dataset_yaml),
            "syn_sample": 1,
            "seed": 46,
            "device": "6",
            "class_subset": {"enabled": False},
            "generation": {
                "model": "/models/gligen",
                "variant": None,
                "local_files_only": True,
                "device": "cuda:0",
                "include_class_name": None,
                "max_objects_per_image": 20,
                "width": 512,
                "height": 512,
                "strength": 0.65,
                "noise_timestep": None,
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
                "gligen_scheduled_sampling_beta": 0.3,
                "global_prompt_source": "template",
                "global_prompt_template": "{phrases}, photo",
                "negative_prompt": "bad image",
                "enable_generation_filter": False,
                "save_visualization": False,
            },
        }
        state_path = self.root / "state.json"
        state = {
            "schema_version": 2,
            "config": cfg,
            "current": {"round": 1, "stage": "generate", "status": "pending"},
            "rounds": {},
            "status": "running",
        }
        runner.write_json_atomic(state_path, state)
        commands: list[list[str]] = []

        def fake_run(command, _cfg, quiet_stage_logs=True):
            commands.append(command)
            output = self.workspace / "round_1/synthetic"
            output.mkdir(parents=True, exist_ok=True)
            (output / "manifest.json").write_text(
                json.dumps([{"image_id": 7}]), encoding="utf-8"
            )

        with patch.object(runner, "run_command", side_effect=fake_run):
            runner.run_generate(state, state_path, cfg, 1)

        command = commands[0]
        self.assertIn("detection_gligen_cos.generate_gligen_sdedit_examples", command)
        self.assertIn("--dataset-yaml", command)
        self.assertIn("--state-json", command)

    def test_internal_retrain_cli_no_longer_accepts_group(self):
        with patch(
            "sys.argv",
            [
                "retrain_runner",
                "--internal-retrain",
                "--state-json",
                str(self.root / "state.json"),
                "--retrain-round",
                "1",
            ],
        ):
            args = runner.parse_args()
        self.assertTrue(args.internal_retrain)
        self.assertFalse(hasattr(args, "retrain_group"))

    def test_run_retrain_launches_one_cos_prompt_training(self):
        cfg = {
            "dataset": str(self.dataset_yaml),
            "workspace": str(self.workspace),
            "device": "cpu",
            "model": {"ultralytics_root": str(self.root / "ultralytics")},
            "retrain": {
                "project": str(self.root / "runs"),
                "run_name_template": "round_{round}_cos_prompt",
            },
        }
        state_path = self.root / "retrain-state.json"
        state = {
            "schema_version": 2,
            "config": cfg,
            "current": {"round": 1, "stage": "retrain", "status": "pending"},
            "rounds": {},
            "status": "running",
        }
        runner.write_json_atomic(state_path, state)
        commands: list[list[str]] = []

        def fake_run(command, _cfg, quiet_stage_logs=True):
            commands.append(command)
            best = self.root / "runs/round_1_cos_prompt/weights/best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.touch()

        with patch.object(runner, "run_command", side_effect=fake_run):
            runner.run_retrain(state, state_path, cfg, 1)

        self.assertEqual(len(commands), 1)
        self.assertIn("detection_gligen_cos.retrain_runner", commands[0])
        self.assertNotIn("--retrain-group", commands[0])
        completed = runner.read_json(state_path)["rounds"]["1"]["retrain"]
        self.assertEqual(completed["status"], "completed")
        self.assertNotIn("groups", completed)

    def test_run_optimize_invokes_cos_module_with_alignment(self):
        model = self.root / "model.pt"
        model.touch()
        gradient = {
            "enabled": True,
            "guide_root": None,
            "guide_split": "val",
            "image_size": 640,
            "guide_batch_size": 2,
            "workers": 0,
            "parameter_scope": "detection_head",
            "implementation": "exact",
            "loss": "hinge",
            "weight": 1.0,
            "margin": 0.0,
            "warmup_steps": 1,
            "epsilon": 1e-12,
            "max_objects_per_image": 20,
            "min_bbox_area_ratio": 0.0,
            "scaleup": False,
            "cache": True,
            "enable_layout_filter": False,
            "max_guide_images": None,
        }
        optimization = {
            "max_train_steps": 2,
            "resume_from_previous": False,
            "prompt_scope": "class",
            "num_new_tokens": 4,
            "initializer_token": "object",
            "init_mode": "class_name",
            "resume_mode": "overwrite",
            "resolution": 512,
            "learning_rate": 5e-4,
            "strength": 0.65,
            "num_inference_steps": 2,
            "semantic_weight": 1.0,
            "detector_weight": 0.1,
            "class_semantic_weight": 0.0,
            "class_semantic_model": "clip",
            "save_steps": 1,
            "log_every": 1,
            "gradient_alignment": gradient,
        }
        cfg = {
            "workspace": str(self.workspace),
            "dataset": str(self.dataset_yaml),
            "device": "cpu",
            "seed": 1,
            "class_subset": {"enabled": False},
            "model": {
                "family": "yolo",
                "ultralytics_root": str(self.root / "ultralytics"),
                "image_size": 640,
                "inference": {"iou": 0.7, "max_det": 300},
            },
            "stable_diffusion": {"model": "sd", "local_files_only": True},
            "prompt_optimization": optimization,
        }
        state_path = self.root / "optimize-state.json"
        state = {
            "schema_version": 2,
            "config": cfg,
            "current": {"round": 1, "stage": "optimize", "status": "pending"},
            "rounds": {},
            "status": "running",
        }
        runner.write_json_atomic(state_path, state)
        commands: list[list[str]] = []

        def fake_run(command, _cfg, quiet_stage_logs=True):
            commands.append(command)
            prompts = self.workspace / "round_1/prompts"
            prompts.mkdir(parents=True, exist_ok=True)
            (prompts / "learned_embeds-2.bin").touch()
            (prompts / "object_prompts.json").write_text(
                json.dumps(
                    {
                        "latest": "learned_embeds-2.bin",
                        "gradient_alignment": {
                            "guide_metadata_hash": "hash",
                            "guide_cache": {"cache_path": "cache.pt"},
                            "statistics": {"overall": {"count": 2}},
                        },
                    }
                ),
                encoding="utf-8",
            )

        with patch.object(runner, "run_command", side_effect=fake_run):
            runner.run_optimize(state, state_path, cfg, 1, model)

        command = commands[0]
        self.assertIn("detection_gligen_cos.optimize_object_prompts_ultralytics", command)
        self.assertIn("--alignment-enabled", command)
        self.assertIn("--guide-images", command)
        self.assertIn("--alignment-implementation", command)
        completed = runner.read_json(state_path)["rounds"]["1"]["optimize"]
        self.assertEqual(completed["guide_metadata_hash"], "hash")

    def test_model_for_next_round_uses_single_retrain_best(self):
        best = self.root / "best.pt"
        best.touch()
        state = {
            "rounds": {
                "1": {
                    "retrain": {"status": "completed", "best": str(best)}
                }
            }
        }
        self.assertEqual(
            runner.model_for_round(state, {"baseline": "unused"}, 2), best
        )


if __name__ == "__main__":
    unittest.main()
