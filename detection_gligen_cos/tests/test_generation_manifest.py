import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from detection_gligen_cos import retrain_runner as runner

from detection_gligen_cos.generation_manifest import (
    append_record, export_manifest, initialize_journal, load_manifest,
    progress_count, recover_manifest, write_json_atomic,
)


class GenerationManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.first = {"image_id": 1, "result_file": "images/one.jpg", "prompt": "测试"}
        self.second = {"image_id": 2, "result_file": "images/two.jpg"}

    def test_append_preserves_history_and_exports_original_format(self):
        initialize_journal(self.output, [], resume=False)
        append_record(self.output, self.first)
        journal = self.output / "manifest.jsonl"
        prefix = journal.read_bytes()
        append_record(self.output, self.second)
        self.assertTrue(journal.read_bytes().startswith(prefix))
        self.assertFalse((self.output / "manifest.json").exists())
        records = load_manifest(self.output)
        self.assertEqual(records, [self.first, self.second])
        export_manifest(self.output, records)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), records)
        self.assertEqual(json.loads((self.output / "selected_images.json").read_text()),
                         [str((self.output / item["result_file"]).resolve()) for item in records])

    def test_torn_tail_is_ignored_then_removed_before_append(self):
        journal = self.output / "manifest.jsonl"
        journal.write_bytes((json.dumps(self.first) + "\n").encode() + b'{"prompt":"\xe6')
        before = journal.read_bytes()
        self.assertEqual(load_manifest(self.output), [self.first])
        self.assertEqual(journal.read_bytes(), before)
        self.assertEqual(load_manifest(self.output, repair=True), [self.first])
        append_record(self.output, self.second)
        self.assertEqual(load_manifest(self.output), [self.first, self.second])

    def test_complete_corrupt_record_is_not_silently_discarded(self):
        (self.output / "manifest.jsonl").write_text('{bad}\n')
        with self.assertRaises(json.JSONDecodeError):
            load_manifest(self.output, repair=True)

    def test_legacy_migration_and_journal_overrides_stale_snapshot(self):
        export_manifest(self.output, [self.first])
        records = load_manifest(self.output, repair=True)
        initialize_journal(self.output, records, resume=True)
        append_record(self.output, self.second)
        initialize_journal(self.output, [], resume=True)
        self.assertEqual(load_manifest(self.output), [self.first, self.second])
        recover_manifest(self.output)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()),
                         [self.first, self.second])

    def test_recovery_exports_journal_when_no_snapshot_exists(self):
        initialize_journal(self.output, [self.first], resume=False)
        recover_manifest(self.output)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), [self.first])
        initialize_journal(self.output, [], resume=False)
        self.assertEqual(load_manifest(self.output), [])

    def test_polling_reads_small_progress_file_and_legacy_fallback(self):
        self.assertEqual(progress_count(self.output), 0)
        export_manifest(self.output, [self.first])
        self.assertEqual(progress_count(self.output), 1)
        write_json_atomic(self.output / "progress.json", {"generated_samples": 2})
        (self.output / "manifest.json").write_text("invalid old snapshot")
        self.assertEqual(progress_count(self.output), 2)

    def test_runner_recovers_journal_before_completion_and_resume_decisions(self):
        for target in (1, 2):
            with self.subTest(target=target):
                workspace = self.output / str(target)
                output = workspace / "round_1/synthetic"
                output.mkdir(parents=True)
                initialize_journal(output, [self.first], resume=False)
                cfg = {"workspace": str(workspace), "syn_sample": target, "device": "cpu",
                       "generation": {"devices": "cpu"}}
                state = {"schema_version": 2, "config": cfg, "rounds": {}}
                runner.round_state(state, 1)["generate"]["status"] = "running"
                state_path = workspace / "state.json"

                def finish(*args, **kwargs):
                    export_manifest(output, [self.first, self.second])

                with patch.object(runner, "data_paths", return_value={}), \
                        patch.object(runner, "build_generation_command", return_value=[]) as build, \
                        patch.object(runner, "run_command", side_effect=finish) as run:
                    runner.run_generate(state, state_path, cfg, 1)
                if target == 1:
                    run.assert_not_called()
                else:
                    self.assertTrue(build.call_args.kwargs["resume"])
                    run.assert_called_once()
                self.assertEqual(state["rounds"]["1"]["generate"]["status"], "completed")
                self.assertEqual(len(json.loads((output / "manifest.json").read_text())), target)


if __name__ == "__main__":
    unittest.main()
