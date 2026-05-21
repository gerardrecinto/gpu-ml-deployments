import argparse
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch GPU training demo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = nn.Sequential(
        nn.Linear(args.input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    torch.manual_seed(42)
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


if __name__ == "__main__":
    main()
