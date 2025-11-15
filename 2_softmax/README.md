# Softmax
Softmax kernel from [official tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py)

## Walkthrough
See `main.py`.

It has plenty of comments to make you understand.

If you have any questions, feel free to leave comment on dedicated notion page.

## How to run
```bash
python3 main.py
```

## Take-home message
- With respect to `BLOCK_SIZE`:
  - It has to be power-of-two (mask redundant elements)
  - Sometimes it has to be fixed (for example in this case: each kernel computes row)
- All we need to consider is up to 2D in most cases
  - Tensors with >2 dimensions can be handled by `view`ing it in 2D
