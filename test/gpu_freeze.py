import torch
import math

def main(
    matmul_size: int = 6144,   # a bit smaller than 8192
    chunk_mb: int = 256,       # smaller chunks
    max_chunks: int = 16       # hard limit so we don't fill EVERYTHING
):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS (Metal) backend not available.")

    device = torch.device("mps")
    print(f"Using device: {device}")
    print(f"Matmul size: {matmul_size} x {matmul_size}")
    print(f"Filler chunk size: {chunk_mb} MB, max_chunks={max_chunks}")

    # Allocate matmul tensors first
    print("Allocating matmul tensors on MPS...")
    n = matmul_size
    try:
        A = torch.randn((n, n), device=device, dtype=torch.float32)
        B = torch.randn((n, n), device=device, dtype=torch.float32)
    except RuntimeError as e:
        raise SystemExit(
            f"Failed to allocate matmul tensors of size {n}x{n}.\n"
            f"Try reducing matmul_size.\nError: {e}"
        )

    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    # Allocate filler tensors
    filler_tensors = []
    bytes_per_elem = 4
    chunk_bytes = chunk_mb * 1024**2
    elems_per_chunk = chunk_bytes // bytes_per_elem

    print("Allocating filler tensors (but with a hard cap)...")
    allocated_chunks = 0
    while allocated_chunks < max_chunks:
        try:
            t = torch.zeros(elems_per_chunk, device=device, dtype=torch.float32)
            filler_tensors.append(t)
            allocated_chunks += 1
            print(f"  -> allocated {allocated_chunks} chunks of {chunk_mb} MB "
                  f"(~{allocated_chunks * chunk_mb} MB total)")
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except RuntimeError as e:
            print("Hit allocation limit earlier than expected (RuntimeError).")
            print(f"Last error: {e}")
            break

    print("\nSetup complete. Starting heavy matmul loop...")
    print("Press Ctrl+C to stop.\n")

    it = 0
    try:
        while True:
            C = A @ B
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            it += 1
            if it % 5 == 0:
                print(f"Iterations: {it}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up and exiting...")
    finally:
        del A, B, filler_tensors
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        print("Cleanup done.")

if __name__ == "__main__":
    main(matmul_size=6144, chunk_mb=256, max_chunks=70)