from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from detection_gligen_sdedit import generate_gligen_sdedit_examples as generator
from detection_gligen_sdedit.generate_hard_examples import choose_group


def parse_experiment_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--phrase-mode",
        required=True,
        choices=["class", "learned", "class_learned"],
        help="Text used for each GLIGEN box.",
    )
    parser.add_argument(
        "--transfer-source-class-id",
        type=int,
        default=None,
        help="Use learned tokens from this source class for every target box.",
    )
    return parser.parse_known_args(argv)


def make_phrase_builder(phrase_mode: str, transfer_source_class_id: int | None):
    def phrase_for_ann(meta: dict, dataset, ann: dict, prompt_scope: str,
                       include_class_name: bool) -> tuple[str, str | None, str]:
        target_class_id = int(ann["class_id"])
        target_class_name = dataset.class_names[target_class_id]
        if phrase_mode == "class":
            return target_class_name, None, "class_prompt"

        token_class_id = target_class_id if transfer_source_class_id is None else transfer_source_class_id
        group = choose_group(meta["groups"], token_class_id, prompt_scope)
        tokens = meta["groups"].get(group)
        if not tokens:
            if transfer_source_class_id is not None:
                raise KeyError(
                    f"No learned prompt group for transfer class_id={token_class_id}; "
                    f"available groups={sorted(meta['groups'])}"
                )
            return target_class_name, None, "class_name_fallback_no_learned_prompt"

        token_text = ",".join(tokens)
        transfer_suffix = "_transfer" if transfer_source_class_id is not None else ""
        if phrase_mode == "learned":
            return token_text, group, f"learned_prompt_only{transfer_suffix}"
        return f"{target_class_name}, {token_text}", group, f"class_plus_learned{transfer_suffix}"

    return phrase_for_ann


def patch_manifest_metadata(output_dir: Path, phrase_mode: str, transfer_source_class_id: int | None) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest:
        item["prompt_difficulty_condition"] = phrase_mode
        item["transfer_source_class_id"] = transfer_source_class_id
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "phrase_mode": phrase_mode,
        "transfer_source_class_id": transfer_source_class_id,
        "paired_generation_note": (
            "Use identical annotations, max-images, seed, layout/filter settings and diffusion settings "
            "across conditions for a controlled comparison."
        ),
    }
    (output_dir / "prompt_condition.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def option_value(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise ValueError(f"Missing value after {name}")
    return argv[index + 1]


def main() -> None:
    experiment_args, generator_argv = parse_experiment_args(sys.argv[1:])
    output_dir_value = option_value(generator_argv, "--output-dir")
    if output_dir_value is None:
        raise ValueError("--output-dir is required by the base GLIGEN-SDEdit generator")

    generator.phrase_for_ann = make_phrase_builder(
        experiment_args.phrase_mode,
        experiment_args.transfer_source_class_id,
    )
    sys.argv = [sys.argv[0], *generator_argv]
    generator.main()
    patch_manifest_metadata(
        Path(output_dir_value),
        experiment_args.phrase_mode,
        experiment_args.transfer_source_class_id,
    )


if __name__ == "__main__":
    main()
