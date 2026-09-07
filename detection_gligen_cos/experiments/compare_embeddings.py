from pathlib import Path
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt1", required=True)
    parser.add_argument("--ckpt2", required=True)
    args = parser.parse_args()

    emb1 = torch.load(args.ckpt1, map_location="cpu")
    emb2 = torch.load(args.ckpt2, map_location="cpu")

    keys1 = set(emb1.keys())
    keys2 = set(emb2.keys())

    print(f"ckpt1 tokens: {len(keys1)}")
    print(f"ckpt2 tokens: {len(keys2)}")

    common = sorted(keys1 & keys2)

    if len(common) == 0:
        print("No common tokens found.")
        return

    distances = []

    for token in common:
        e1 = emb1[token].float()
        e2 = emb2[token].float()

        l2 = torch.norm(e1 - e2).item()
        cos = torch.nn.functional.cosine_similarity(
            e1.unsqueeze(0),
            e2.unsqueeze(0)
        ).item()

        distances.append(l2)

        print(
            f"{token}: "
            f"L2={l2:.6f}, "
            f"cos={cos:.6f}"
        )

    print("\nSummary")
    print(f"tokens compared: {len(distances)}")
    print(f"mean L2: {sum(distances)/len(distances):.6f}")
    print(f"max L2: {max(distances):.6f}")
    print(f"min L2: {min(distances):.6f}")


if __name__ == "__main__":
    main()