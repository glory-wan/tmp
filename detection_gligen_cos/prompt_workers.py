"""Class-partitioned Prompt optimization with a shared, read-only Guide cache."""

import json
import os
import subprocess
import time
from pathlib import Path


def allocate_workers(classes, devices, total_steps):
    if total_steps <= 0 or not classes:
        raise ValueError("Prompt optimization requires positive steps and nonempty classes")
    # With tiny budgets, use fewer GPUs so every worker gets at least one step.
    count = min(len(devices), len(classes), total_steps)
    partitions = [sorted(classes)[index::count] for index in range(count)]
    budgets = [total_steps * len(part) // len(classes) for part in partitions]
    order = sorted(range(count), key=lambda i: -(total_steps * len(partitions[i]) % len(classes)))
    for index in order[:total_steps - sum(budgets)]:
        budgets[index] += 1
    return [{"classes": part, "device": devices[i], "steps": budgets[i]}
            for i, part in enumerate(partitions)]


def scaled_steps(value, local_steps, total_steps):
    return max(1, round(value * local_steps / total_steps)) if value else 0


def replace_options(command, **options):
    command = list(command)
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if flag in command:
            command[command.index(flag) + 1] = str(value)
        else:
            command.extend([flag, str(value)])
    return command


def merge_workers(prompts, workers, total_steps, warmup_steps):
    import torch
    from .prompt_gradient_alignment import AlignmentRunningStatistics
    from .generate_gligen_sdedit_examples import write_json_atomic

    active = {f"class_{cls}" for worker in workers for cls in worker["classes"]}
    groups, embeddings, statistics = {}, {}, {}
    metadata = None
    guide_hash = None
    for worker in workers:
        output = Path(worker["output"])
        info = json.loads((output / "object_prompts.json").read_text(encoding="utf-8"))
        expected = {f"class_{cls}" for cls in worker["classes"]}
        if set(info["train_groups"]) != expected:
            raise RuntimeError(f"Prompt worker class assignment mismatch: {output}")
        if info["latest"] != f"learned_embeds-{worker['steps']}.bin":
            raise RuntimeError(f"Prompt worker has not completed its step budget: {output}")
        learned = torch.load(output / info["latest"], map_location="cpu", weights_only=True)
        current_hash = info.get("gradient_alignment", {}).get("guide_metadata_hash")
        if metadata is None:
            metadata, guide_hash = info, current_hash
        elif guide_hash != current_hash:
            raise RuntimeError("Prompt workers used different Guide gradients")
        for group, tokens in info["groups"].items():
            # A worker may carry previous-round copies of other workers' tokens.
            if group in active and group not in expected:
                continue
            if group in groups:
                if groups[group] != tokens or any(not torch.equal(embeddings[t], learned[t]) for t in tokens):
                    raise RuntimeError(f"Inconsistent preserved Prompt group: {group}")
                continue
            groups[group] = tokens
            for token in tokens:
                if token in embeddings:
                    raise RuntimeError(f"Duplicate Prompt token: {token}")
                embeddings[token] = learned[token]
        stats = info.get("gradient_alignment", {}).get("statistics", {}).get("groups", {})
        statistics.update({group: value for group, value in stats.items() if group in expected})
    if not active.issubset(groups):
        raise RuntimeError("Missing trained Prompt groups during merge")
    metadata = dict(metadata)
    # Each worker has its own scheduler timeline; there is no merged optimizer.
    metadata.pop("lr_scheduler_state", None)
    metadata.update(groups=groups, train_groups=sorted(active),
                    preserved_resume_groups=sorted(set(groups) - active),
                    latest=f"learned_embeds-{total_steps}.bin", workers=workers,
                    completed_steps=total_steps)
    alignment = dict(metadata.get("gradient_alignment", {}))
    alignment["warmup_steps"] = warmup_steps
    alignment["statistics"] = AlignmentRunningStatistics({"groups": statistics}).to_dict()
    metadata["gradient_alignment"] = alignment
    destination = prompts / metadata["latest"]
    temporary = destination.with_suffix(".bin.tmp")
    torch.save(embeddings, temporary)
    os.replace(temporary, destination)
    write_json_atomic(prompts / "object_prompts.json", metadata)


def run_prompt_workers(command, cfg, prompts, devices, initial_step):
    from . import retrain_runner as runner
    from .generate_gligen_sdedit_examples import write_json_atomic

    optimization = cfg["prompt_optimization"]
    if optimization["prompt_scope"] != "class":
        raise ValueError("Multi-GPU Prompt optimization requires prompt_scope: class")
    if initial_step:
        raise ValueError("Cannot repartition an unfinished single-GPU Prompt checkpoint; finish it using one device first")
    total = int(optimization["max_train_steps"])
    gradient = optimization["gradient_alignment"]
    root = prompts / "_workers"
    prepare = root / "prepare"
    prep_command = replace_options(command, output_dir=prepare)
    runner.run_command(prep_command + ["--prepare-workers"], cfg, physical_devices=devices[0])
    classes = json.loads((prepare / "worker_classes.json").read_text(encoding="utf-8"))
    workers = allocate_workers(classes, devices, total)
    for index, worker in enumerate(workers):
        worker["output"] = str(root / f"worker_{index}")
        worker["lr_warmup_steps"] = scaled_steps(int(optimization.get("lr_warmup_steps", 0)), worker["steps"], total)
        worker["alignment_warmup_steps"] = scaled_steps(int(gradient["warmup_steps"]), worker["steps"], total)
    plan = {"workers": workers, "total_steps": total, "command": command}
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError("Prompt worker configuration changed; resume using the original configuration")
    write_json_atomic(plan_path, plan)
    processes = []
    try:
        for index, worker in enumerate(workers):
            output = Path(worker["output"])
            checkpoint, step = runner.latest_prompt_checkpoint(output)
            if not (output / "object_prompts.json").is_file():
                checkpoint, step = None, 0
            if step > worker["steps"]:
                raise RuntimeError(f"Prompt worker checkpoint exceeds its budget: {output}")
            if checkpoint is not None and step == worker["steps"]:
                continue
            local = replace_options(
                command, output_dir=output, train_class_ids=",".join(map(str, worker["classes"])),
                max_train_steps=worker["steps"], initial_step=step,
                alignment_warmup_steps=worker["alignment_warmup_steps"],
                lr_warmup_steps=worker["lr_warmup_steps"],
                save_steps=scaled_steps(int(optimization["save_steps"]), worker["steps"], total),
                seed=int(cfg["seed"]) + index,
            )
            if checkpoint is not None:
                local = replace_options(local, resume_token=checkpoint, resume_mode="overwrite")
            if gradient.get("enabled", True):
                local += ["--alignment-cache", "--require-guide-cache"]
            print(f"Prompt worker {index}: GPU={worker['device']} classes={worker['classes']} steps={step}/{worker['steps']}", flush=True)
            processes.append(subprocess.Popen(
                local, cwd=runner.REPO_ROOT,
                env=runner.subprocess_environment(cfg, True, worker["device"]),
            ))
        while processes:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Prompt worker failed: exit codes={codes}; checkpoints are preserved")
            if all(code is not None for code in codes):
                break
            time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
    merge_workers(prompts, workers, total, int(gradient["warmup_steps"]))
