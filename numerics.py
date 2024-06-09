def clamp(x, min_x, max_x):
    if max_x < x:
        return max_x
    if min_x > x:
        return min_x

    return x

# A simple algorithm that can bisect
# a function
def bisect(f, eps=1e-4):
    x = 0.5
    while True:
        fx = f(x)
        if fx < 0:
            nx = (x + 1) / 2
        elif fx >= 0:
            nx = x / 2
        else:
            return x

        if abs(nx - x) <= eps:
            return x

        x = nx
