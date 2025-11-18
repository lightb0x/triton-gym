def generate_range(start, num_steps, until):
    """
    generates floating-point-like range
    """
    current = start
    step = start
    arr = []
    while current < until:
        arr.append(current)
        current += step
        if current == step * num_steps:
            step *= 2

    return arr


if __name__ == "__main__":
    print(generate_range(128, 8, 16384 + 1))
    """
    [
        (0,) 128, 256, 384, 512, 640, 768, 896,  # 8-1 elements with stepsize=128
        1024, 1280, 1536, 1792,  # 4 elements with stepsize=256
        2048, 2560, 3072, 3584,  # 4 elements with stepsize=512
        4096, 5120, 6144, 7168,  # 4 elements with stepsize=1024
        8192, 10240, 12288, 14336, 16384  # 4+1 elements with stepsize=2048
    ]
    """
