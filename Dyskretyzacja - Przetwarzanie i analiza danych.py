import numpy as np
from numpy.lib import stride_tricks
from math import log

def konfuzja(TN, TP, FN, FP):
    print("TN:", TN)
    print("TP:", TP)
    print("FN:", FN)
    print("FP:", FP)
    print("\n")

    n = TN + TP + FN + FP

    # print('Procent chorych:', (TP + FN) / n * 100, '%')
    # print('Procent zdrowych:', (TN + FP) / n * 100, '%')
    # print('-------------')

    print('Dokładność klasyfikacji:', (TN + TP) / (TN + TP + FN + FP))
    print('Błąd klasyfikacji:', (FN + FP) / (TN + TP + FN + FP))
    print('Czułość:', TP / (TP + FN))
    print('Specyficzność:', TN / (TN + FP))
    print('Precyzja:', TP / (TP + FP))
    print('F1:', 2 * TP / (2 * TP + FP + FN))
    print('Zbalansowana dokładność:', 1 / 2 * (TP / (TP + FN)) + 1 / 2 * (TN / (TN + FP)))
    print('-------------')

    # print("Prawdopodobieństwo poprawnego wykrycia choroby, wiedząc, że klasyfikator podjął decyzję o chorobie:",
    #       TP / (TP + FP))
    # print('Prawdopodobieństwo poprawnego wskazania osoby zdrowej, wiedząc, że klasyfikator podjął decyzję o braku choroby:',
    #       TN / (TN + FN))
    # print('Prawdopodobieństwo niepoprawnego wykrycia choroby, wiedząc, że klasyfikator podjął decyzję o chorobie:',
    #       FP / (TP + FP))
    # print('Prawdopodobieństwo niepoprawnego wskazania osoby zdrowej, wiedząc, że klasyfikator podjął decyzję o braku choroby:',
    #       FN / (TN + FN))

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


def dyskretyzacja(dane, prog):
    print("Próg dyskretyzacji: ", prog)

    if len(dane.shape) == 1:
        elemSize = dane.strides[0]
        dane = stride_tricks.as_strided(dane, (len(dane) // 2, 2), (2 * elemSize, elemSize))

        lt_0 = (dane[:, 0] == 0) & (dane[:, 1] < prog)
        lt_1 = (dane[:, 0] != 0) & (dane[:, 1] < prog)
        ge_0 = (dane[:, 0] == 0) & (dane[:, 1] >= prog)
        ge_1 = (dane[:, 0] != 0) & (dane[:, 1] >= prog)

        emp = np.array([
            [np.count_nonzero(lt_0), np.count_nonzero(lt_1)],
            [np.count_nonzero(ge_0), np.count_nonzero(ge_1)]
        ])

        n = emp.sum()

        print("n = ", n)
        print("Rozkład empiryczny: ")
        print(emp, '\n')
        print("Prawdopodobieństwo łączne: ")
        print(str(emp[0,0]) + "/" + str(n) + ", " + str(emp[0,1]) + "/" + str(n))
        print(str(emp[1, 0]) + '/' + str(n) + ', ' + str(emp[1, 1]) + '/' + str(n))

        prawd = emp / n
        I = wyniki(prawd)

        TN, FN, FP, TP = emp[0, 0], emp[0, 1], emp[1, 0], emp[1, 1]
        konfuzja(TN, TP, FN, FP)

        return

tabela_z2 = np.array(
[1, 0.75,
 0, 0.85,
 1, 0.34,
 1, 0.71,
 0, 0.5,
 0, 0.55,
 0, 0.43,
 1, 0.37,
 1, 0.2,
 0, 0.1]
)

dyskretyzacja(tabela_z2, 0.6)