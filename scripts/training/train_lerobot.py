import argparse
import torch
from pathlib import Path

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features


def main(args):
    output_directory = Path(args.checkpoint)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    training_steps = args.training_steps
    log_freq = args.log_freq

    dataset_metadata = LeRobotDatasetMetadata(args.repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if args.model_type == 'diffusion':
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        cfg = DiffusionConfig(
            n_obs_steps=args.n_obs_steps,
            input_features=input_features, 
            output_features=output_features,
            push_to_hub=False,
            horizon=args.chunk_size,
            n_action_steps=args.n_action_steps,
        )
        policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    elif args.model_type == 'act':
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        cfg = ACTConfig(
            n_obs_steps=args.n_obs_steps,
            input_features=input_features, 
            output_features=output_features,
            push_to_hub=False,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
        )
        policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    elif args.model_type == 'smolvla':
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        cfg = SmolVLAConfig(
            n_obs_steps=args.n_obs_steps,
            input_features=input_features, 
            output_features=output_features,
            push_to_hub=False,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            load_vlm_weights=True,
        )
        policy = SmolVLAPolicy(cfg, dataset_stats=dataset_metadata.stats)
    else:
        raise NotImplementedError

    policy.train()
    policy.to(device)

    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    if cfg.observation_delta_indices is not None:
        delta_timestamps["observation.state"] = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
        for key in features.keys():
            if key.startswith('observation.image'):
                delta_timestamps[key] = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
    
    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LeRobot policy.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="diffusion",
        choices=["diffusion", "act", "smolvla"],
        help="Type of model to train.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/lerobot",
        help="Directory to save the policy checkpoint.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Koorye/pika-example",
        help="Repository ID of the LeRobot dataset.",
    )
    parser.add_argument(
        "--n-obs-steps",
        type=int,
        default=2,
        help="Number of observation steps to use in the policy.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Horizon for the policy.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Number of action steps to use in the policy.",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1000,
        help="Number of training steps to run.",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=1,
        help="Frequency of logging training progress (in steps).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for the DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use for training (e.g., 'cuda:0' or 'cpu').",
    )
    args = parser.parse_args()
    main(args)
