# Vector adder module
computes `relu(A + B)` or `max(A + B, 0)` elementwise

## Outline
- `add.py` : PyTorch module
- `add_fwd.py` : forward kernel
- `add_bwd.py` : backward kernel
- `test_and_bench.py` : validate and perform benchmarks

## How to run
```bash
python3 add.py
```

## Take-home message
- Triton compiler auto-tunes GPU parameters just-in-time (JIT)
- Best practice is to modularize
  - main
  - fwd
  - bwd
  - test
  - benchmark
- gradient (`grad_output`, when used as input of Triton kernel) need to be copied and/or contiguous
- basic `torch.compile` does job well for normal workflows
