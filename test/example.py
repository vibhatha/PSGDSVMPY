def test(x,z):
    y = 0
    for i in range(x):
        for j in range(z):
            y += i+j
            y = y * 10 - 11
            y = x + z
            y = y*z -x
    return y
