"""Print PyTorch CUDA diagnostics for AVISTA."""

from __future__ import annotations


def main() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"torch version: unavailable ({exc})")
        print("cuda available: False")
        print("torch cuda version: unavailable")
        print("gpu count: 0")
        print("gpu name: unavailable")
        print("tensor test result: failed")
        return

    cuda_available = torch.cuda.is_available()
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {cuda_available}")
    print(f"torch cuda version: {torch.version.cuda}")
    print(f"gpu count: {torch.cuda.device_count() if cuda_available else 0}")

    if not cuda_available:
        print("gpu name: unavailable")
        print("tensor test result: skipped")
        return

    try:
        print(f"gpu name: {torch.cuda.get_device_name(0)}")
        tensor = torch.tensor([1.0], device="cuda")
        print(f"tensor test result: {bool(tensor.item() == 1.0)}")
    except Exception as exc:
        print("gpu name: unavailable")
        print(f"tensor test result: failed ({exc})")


if __name__ == "__main__":
    main()
