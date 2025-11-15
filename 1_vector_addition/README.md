# Vector addition
Basic 1D kernel from [official tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py).

## Walkthrough
See `main.py`.

It has plenty of comments to make you understand.

If you have any questions, feel free to leave comment on dedicated notion page.

## How to run
```bash
python3 main.py
```

## Take-home message
- Triton programming model simplifies memory down to *GMEM (global memory)* and *SMEM (shared memory)*
- Every Triton kernel need to be *validated* and *benchmarked*
