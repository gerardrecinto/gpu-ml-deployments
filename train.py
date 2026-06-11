import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch GPU training demo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-model", default=None, help="Save trained model weights to path")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} (seed={args.seed})")

    model = nn.Sequential(
        nn.Linear(args.input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    data = torch.randn(args.batch_size, args.input_dim, device=device)
    target = torch.randn(args.batch_size, 1, device=device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print("-" * 40)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch:2d}/{args.epochs}]  loss: {loss.item():.6f}")

    print("-" * 40)
    print("Training complete.")

    if args.output_model:
        out_dir = os.path.dirname(args.output_model)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), args.output_model)
        print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
