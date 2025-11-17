def is_power_of_two(x: int):
    return (x & (x - 1)) == 0
