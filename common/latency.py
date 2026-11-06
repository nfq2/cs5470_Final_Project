"""Model latency measured one way: batch 1, inputs already on the GPU.

    forward_latency(lambda: model(images, history))

Excludes file reading, preprocessing and host-to-GPU transfer; a branch that also reports
end-to-end latency times those separately. Run with FP32 and TF32 disabled unless the
experiment is about precision.
"""
import statistics
import time

WARMUP = 20
REPEATS = 100


def forward_latency(forward, warmup=WARMUP, repeats=REPEATS):
    import torch

    times = []
    with torch.no_grad():
        for _ in range(warmup):  # Warm up the GPU first.
            forward()
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            forward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
            "warmup": warmup, "repeats": repeats}
