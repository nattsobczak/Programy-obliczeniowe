import numpy as np
from math import log


def n2p(n):
    n = np.array(n)
    suma = np.sum(n)
    p = n / suma
    return(p)


def wyniki(p, X_gora = True):
    p = np.array(p)
    if X_gora == True:
        px = np.sum(p, axis = 0)
        py = np.sum(p, axis = 1)
    else:
        px = np.sum(p, axis = 1)
        py = np.sum(p, axis = 0)

    hx = 0
    for prawd in px:
        if prawd != 0:
            temp = prawd * log(1 / prawd, 2)
            hx += temp
    hx = round(hx, 4)


    hy = 0
    for prawd in py:
        if prawd != 0:
            temp = prawd * log(1 / prawd, 2)
            hy += temp
    hy = round(hy, 4)

    hxy = 0
    for row in p:
        for prawd in row:
            if prawd != 0:
                temp = prawd * log(1 / prawd, 2)
                hxy += temp
    hxy = round(hxy, 4)

    I = round(hx + hy - hxy, 4)
    Hy_x = round(hy - I, 4)
    Hx_y = round(hx - I, 4)

    print("H(x) =", hx)
    print("H(y) =", hy)
    print("H(x,y) =", hxy)
    print("H(x|y) =", Hx_y)
    print("H(y|x) =", Hy_x)
    print("I(x,y) =", I)
    print("")

# print("Dane dla częstości")
# n = [[2, 3, 1],
#      [3, 4, 1],
#      [0, 2, 0]]

print("Dane dla prawdopodobieństwa łącznego")
n = [[1/3, 1/3],
     [1/6, 1/6]]

p = n2p(n)
wyniki(p)

