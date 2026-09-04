"""Prove the GPU actually works before writing any training code.

Blackwell is sm_120. PyTorch wheels built for older architectures install
cleanly and then fail at the first kernel launch, so checking
torch.cuda.is_available() proves nothing -- this script runs real kernels,
compiles a real graph, and does a real forward/backward pass.

    python scripts/verify_env.py
"""

import sys
import time

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def ok(msg):
    return True, msg


def fail(msg):
    return False, msg


@check("Python version")
def _python():
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 10):
        return fail(f"{s} -- need 3.10+")
    return ok(s)


@check("PyTorch build")
def _torch_build():
    import torch
    cuda = torch.version.cuda
    if cuda is None:
        return fail(f"{torch.__version__} is a CPU-only build -- reinstall with a CUDA index URL")
    major, minor = (int(x) for x in cuda.split(".")[:2])
    if (major, minor) < (12, 8):
        return fail(f"torch {torch.__version__} / CUDA {cuda} -- Blackwell needs CUDA 12.8+")
    return ok(f"torch {torch.__version__} / CUDA {cuda}")


@check("GPU visible")
def _gpu():
    import torch
    if not torch.cuda.is_available():
        return fail("no CUDA device -- inside WSL check that the Windows driver is 570+")
    props = torch.cuda.get_device_properties(0)
    cap = f"sm_{props.major}{props.minor}"
    vram = props.total_memory / 1024**3
    arch_list = torch.cuda.get_arch_list()
    supported = f"sm_{props.major}{props.minor}" in arch_list
    detail = f"{props.name}, {vram:.1f} GB, {cap}"
    if not supported:
        return fail(f"{detail} -- this build targets {arch_list}, not {cap}")
    return ok(detail)


@check("bf16 matmul")
def _matmul():
    import torch
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    got = (a @ b).float()
    want = a.float() @ b.float()
    torch.cuda.synchronize()
    err = (got - want).abs().max().item()
    if err > 2.0:  # bf16 has ~3 decimal digits; over a 512-term sum this is generous
        return fail(f"max abs error {err:.3f} -- kernels are producing garbage")
    return ok(f"correct (max abs error {err:.3f})")


@check("bf16 throughput")
def _throughput():
    import torch
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = (2 * n**3) / dt / 1e12
    if tflops < 50:
        return fail(f"{tflops:.0f} TFLOP/s -- far below expected, check thermals or driver")
    return ok(f"{tflops:.0f} TFLOP/s dense")


@check("flash SDPA backend")
def _sdpa():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(2, 16, 512, 64, device="cuda", dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
    except RuntimeError as e:
        return fail(f"flash backend unavailable ({e}) -- training will be slower but correct")
    return ok("available (no flash-attn package needed)")


@check("torch.compile")
def _compile():
    import torch
    m = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
    ).cuda()
    try:
        c = torch.compile(m, mode="max-autotune")
        x = torch.randn(8, 256, device="cuda")
        c(x).sum().backward()
        torch.cuda.synchronize()
    except Exception as e:
        return fail(f"{type(e).__name__}: {e} -- costs ~30% throughput, needs a C++ toolchain")
    return ok("compiles and runs")


@check("model forward/backward")
def _model():
    import torch
    sys.path.insert(0, ".")
    from core.config import ModelConfig
    from core.model import Transformer

    cfg = ModelConfig()
    model = Transformer(cfg).cuda().to(torch.bfloat16)
    n = model.num_params()
    if abs(n - cfg.n_params()) > 0:
        return fail(f"built {n:,} params but config predicts {cfg.n_params():,}")

    # Targets must be shifted -- see Transformer.forward. Passing idx unshifted
    # makes the model predict its own input, which the tied embeddings make
    # easy, and the loss lands below ln(vocab) looking like a great start.
    tokens = torch.randint(0, cfg.vocab_size, (4, cfg.context_len + 1), device="cuda")
    _, loss, _ = model(tokens[:, :-1], targets=tokens[:, 1:])
    loss.backward()
    torch.cuda.synchronize()

    expected = torch.log(torch.tensor(float(cfg.vocab_size))).item()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if abs(loss.item() - expected) > 0.5:
        return fail(f"initial loss {loss.item():.2f}, expected ~{expected:.2f} -- init is wrong")
    return ok(f"{n / 1e6:.0f}M params, loss {loss.item():.2f} (~ln vocab), {peak:.1f} GB peak")


def main():
    print()
    width = 26
    failures = 0
    for name, fn in CHECKS:
        try:
            passed, msg = fn()
        except Exception as e:  # a check that explodes is a failed check
            passed, msg = False, f"{type(e).__name__}: {e}"
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}]  {name.ljust(width)}  {msg}")
        if not passed:
            failures += 1
    print()
    if failures:
        print(f"  {failures} check(s) failed. Fix these before starting a training run.\n")
        return 1
    print("  Environment is ready.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
