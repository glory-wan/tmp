import unittest
from itertools import islice
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from detection_gligen_cos import generate_gligen_sdedit_examples as generation


class GenerationBatchTest(unittest.TestCase):
    def test_batch_preserves_per_image_rng_and_cfg_condition_order(self):
        args = SimpleNamespace(device="cpu", width=8, height=8, guidance_scale=7.5,
                               num_inference_steps=2, strength=1., noise_timestep=None,
                               gligen_scheduled_sampling_beta=0.5)
        sources = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
        sizes, conditions = [], []

        class Scheduler:
            def set_timesteps(self, count, device):
                self.timesteps = torch.arange(count, 0, -1)

            def add_noise(self, latents, noise, timesteps):
                return latents + noise

            def scale_model_input(self, latents, timestep):
                return latents

            def step(self, prediction, timestep, latents):
                return SimpleNamespace(prev_sample=latents)

        def encode(images):
            sizes.append(len(images))
            return SimpleNamespace(latent_dist=SimpleNamespace(mean=images, std=torch.ones_like(images)))

        def unet(latents, timestep, encoder_hidden_states, cross_attention_kwargs):
            conditions.append((len(latents), cross_attention_kwargs["gligen"]))
            return SimpleNamespace(sample=torch.zeros_like(latents))

        pipe = SimpleNamespace(
            scheduler=Scheduler(), enable_fuser=lambda enabled: None, unet=unet,
            encode_prompt=lambda prompts, *a, **k: (torch.ones(len(prompts), 1), torch.zeros(len(prompts), 1)),
            vae=SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scaling_factor=1.),
                                encode=encode, decode=lambda latents, **kwargs: (latents,)),
            image_processor=SimpleNamespace(preprocess=lambda image: torch.zeros(1, 3, 8, 8),
                                            postprocess=lambda images, **kwargs: list(images)),
        )

        def grounding(pipe, phrases, boxes, cfg, device):
            self.assertFalse(cfg)
            return {"gligen": {"boxes": torch.tensor(boxes).unsqueeze(0),
                               "positive_embeddings": torch.tensor([[[float(phrases[0])]]]),
                               "masks": torch.ones(1, 1)}}

        with patch.object(generation, "prepare_gligen_grounding", side_effect=grounding):
            for guidance in (7.5, 1.):
                args.guidance_scale = guidance
                conditions.clear()
                batch, info = generation.run_gligen_sdedit(
                    pipe, sources, ["one", "two"], "negative", [["1"], ["2"]],
                    [[[0., 0., 1., 1.]], [[0., 0., .5, .5]]], args,
                    [torch.Generator().manual_seed(seed) for seed in (11, 22)],
                )
                self.assertEqual(sizes[-1], 2)
                self.assertEqual(info["denoise_steps"], 2)
                self.assertEqual(conditions[0][0], 4 if guidance > 1 else 2)
                expected = [0., 0., 1., 1.] if guidance > 1 else [1., 1.]
                self.assertEqual(conditions[0][1]["masks"].flatten().tolist(), expected)
                self.assertEqual(conditions[0][1]["positive_embeddings"].flatten().tolist(),
                                 [1., 2., 1., 2.] if guidance > 1 else [1., 2.])
                for i, seed in enumerate((11, 22)):
                    single, _ = generation.run_gligen_sdedit(
                        pipe, sources[i], "one", "negative", [str(i + 1)], [[0., 0., 1., 1.]],
                        args, torch.Generator().manual_seed(seed),
                    )
                    torch.testing.assert_close(single, batch[i], rtol=0, atol=0)

    def test_tail_batch_keeps_order_seeds_and_remaining_quota(self):
        args = SimpleNamespace(batch_size=2, device="cpu", seed=46, negative_prompt="negative")
        samples = [dict(image_id=i, source=i, global_prompt=str(i), phrases=[str(i)], boxes=[])
                   for i in range(10)]
        batches, seeds = [], []

        def generate(pipe, sources, prompts, negative, phrases, boxes, args, generators):
            batches.append(sources)
            seeds.extend(g.initial_seed() for g in generators)
            return sources, {}

        with patch.object(generation, "run_gligen_sdedit", side_effect=generate):
            results = list(generation.generate_batches(None, islice(iter(samples), 5), args))
        self.assertEqual(batches, [[0, 1], [2, 3], [4]])
        self.assertEqual([image for sample, image, info in results], list(range(5)))
        self.assertEqual(seeds, list(range(46, 51)))


if __name__ == "__main__":
    unittest.main()
