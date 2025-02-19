"""
This module contains functions to compute the DWPT of a time series
We assume that we compute the partial DWPT up to level J
and that the length of the time series is a multiple of 2**J
"""

from math import floor

import numpy as np

from .dwt import compute_AB, get_scaling, get_wavelet


def get_Wjn(h, g, j, n, Wjm1n):
    """
    Compute the DWPT coefficients Wj,2n,t and Wj,2n+1,t at level j
    from the DWPT coefficients Wj-1,n,t at level j - 1

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the DWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the DWPT scaling filter
        type j = integer
        j = Current level of the DWPT decomposition
        type n = integer
        n = Index of the DWPT vector Wj-1,n at level j - 1
        type Wjm1n = 1D numpy array
        Wjm1n = DWPT coefficients Wj-1,n,t at level j - 1
    Output:
        type Wjn1 = 1D numpy array
        Wjn1 = DWPT coefficients Wj,2n,t at level j
        type Wjn2 = 1D numpy array
        Wjn2 = DWPT coefficients Wj,2n+1,t at level j
    """
    assert np.shape(h)[0] == np.shape(g)[0], (
        'Wavelet and scaling filters have different lengths'
    )
    Njm1 = len(Wjm1n)
    assert Njm1 % 2 == 0, 'Length of input vector Wj-1,n is odd'
    assert (n >= 0) and (n <= 2 ** (j - 1) - 1), (
        'Index n must be >= 0 and <= 2 ** (j - 1) - 1'
    )
    Nj = int(Njm1 / 2)
    Wjn1 = np.zeros(Nj)
    Wjn2 = np.zeros(Nj)
    L = np.shape(h)[0]
    if n % 2 == 0:
        an = g
        bn = h
    else:
        an = h
        bn = g
    for t in range(0, Nj):
        for lvl in range(0, L):
            index = int((2 * t + 1 - lvl) % Njm1)
            Wjn1[t] = Wjn1[t] + an[lvl] * Wjm1n[index]
            Wjn2[t] = Wjn2[t] + bn[lvl] * Wjm1n[index]
    return (Wjn1, Wjn2)


def get_Wj(h, g, j, Wjm1):
    """
    Compute the DWPT coefficients Wj at level j
    from the DWPT coefficients Wj-1 at level j - 1

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the DWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the DWPT scaling filter
        type j = integer
        j = Current level of the DWPT decomposition
        type Wjm1 = 1D numpy array
        Wjm1 = DWPT coefficients Wj-1 at level j - 1
    Output:
        type Wj = 1D numpy array
        Wj = DWPT coefficients Wj at level j
    """
    assert np.shape(h)[0] == np.shape(g)[0], (
        'Wavelet and scaling filters have different lengths'
    )
    N = len(Wjm1)
    assert N % (2**j) == 0, (
        'Length of input vector Wj-1 must be a multiple of 2 ** (j - 1)'
    )
    Wj = np.zeros(N)
    Njm1 = int(N / (2 ** (j - 1)))
    Nj = int(N / (2**j))
    for n in range(0, 2 ** (j - 1)):
        Wjm1n = Wjm1[int(n * Njm1) : int((n + 1) * Njm1)]
        (Wjn1, Wjn2) = get_Wjn(h, g, j, n, Wjm1n)
        Wj[int(2 * n * Nj) : int((2 * n + 1) * Nj)] = Wjn1
        Wj[int((2 * n + 1) * Nj) : int(2 * (n + 1) * Nj)] = Wjn2
    return Wj


def get_DWPT(X, name, J):
    """
    Compute the DWPT of X up to level J

    Input:
        type X = 1D numpy array
        X = Time series which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWPT
    Output:
        type W = list of J+1 1D numpy arrays
        W = Vectors of DWPT coefficients at levels 0, ... , J
    """
    assert isinstance(J, int), 'Level of DWPT must be an integer'
    assert J >= 1, 'Level of DWPT must be higher or equal to 1'
    N = np.shape(X)[0]
    assert N % (2**J) == 0, 'Length of time series is not a multiple of 2**J'
    g = get_scaling(name)
    h = get_wavelet(g)
    W = [X]
    for j in range(1, J + 1):
        Wjm1 = W[-1]
        Wj = get_Wj(h, g, j, Wjm1)
        W.append(Wj)
    return W


def compute_c(J):
    """
    Compute index vectors indicating the list of filters (H or G)
    used to compute each of the DWPT coefficients Wj,n

    Input:
        type J = integer
        J = Level of partial DWPT
    Output:
        type c = List of length J
        c = Each element of the list is made of the list of cj,n
            with n = 0, ... , 2 ** j - 1
    """
    assert isinstance(J, int), 'Level of DWPT must be an integer'
    assert J >= 1, 'Level of DWPT must be higher or equal to 1'
    c1 = [[0], [1]]
    c = [c1]
    for j in range(2, J + 1):
        cjm1 = c[-1]
        cj = []
        for n in range(0, int(2**j)):
            index = int(floor(n / 2))
            cjn = cjm1[index].copy()
            if (n % 4 == 0) or (n % 4 == 3):
                cjn.append(0)
            else:
                cjn.append(1)
            cj.append(cjn)
        c.append(cj)
    return c


def compute_basisvector(cjn, name, N):
    """
    Compute the basis vectors of the orthonormal basis
    corresponding to the list cj,n

    Input:
        type cjn = list of length j
        cj,n = 0 if the filter is g / 1 if the filter is h
        type name = string
        name = Name of the wavelet filter
        type N = integer
        N = Length of the time series (multiple of 2**j)
    Output:
        type C = N * N / 2**j numpy array
        C = N / 2**j basis vectors of length N
    """
    g = get_scaling(name)
    h = get_wavelet(g)
    J = len(cjn)
    C = np.identity(int(N / (2**J)))
    for j in range(J, 0, -1):
        if cjn[j - 1] == 0:
            Cj = compute_AB(g, j, N)
        else:
            Cj = compute_AB(h, j, N)
        C = np.matmul(np.transpose(Cj), C)
    return C


def get_nu(name, J):
    """
    Compute the phase shift for LA or coiflet filters

    Input:
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Maximum level for DWPT
    Output:
        type nu = list of J values
        nu = Each element of the list is made of the list of nuj,n
             with n = 0, ... , 2 ** (j - 1)
    """
    assert isinstance(J, int), 'Level of DWPT must be an integer'
    assert J >= 1, 'Level of DWPT must be higher or equal to 1'
    assert name[0:2] == 'LA' or name[0:1] == 'C', (
        'Wavelet filter must be Daubechies least asymmetric or Coiflet'
    )
    c = compute_c(J)
    # Compute Sj,n,1
    S = []
    for j in range(1, J + 1):
        Sj = []
        for n in range(0, 2**j):
            Sjn = 0.0
            cjn = c[j - 1][n]
            for m in range(0, j):
                Sjn = Sjn + cjn[m] * (2**m)
            Sj.append(Sjn)
        S.append(Sj)
    # Compute nuj,n
    nu = []
    for j in range(1, J + 1):
        nuj = []
        for n in range(0, 2**j):
            # Least asymmetric
            if name[0:2] == 'LA':
                L = int(name[2:])
                Lj = int((2**j - 1) * (L - 1) + 1)
                if L == 14:
                    nujn = int(-Lj / 2 + 3 * (2 ** (j - 1) - S[j - 1][n]) - 1)
                elif int(L / 2) % 2 == 0:
                    nujn = int(-Lj / 2 + (2 ** (j - 1) - S[j - 1][n]))
                else:
                    nujn = int(-Lj / 2 - (2 ** (j - 1) - S[j - 1][n]) + 1)
            # Coiflet
            else:
                L = int(name[1:])
                Lj = int((2**j - 1) * (L - 1) + 1)
                nujn = int(
                    -Lj / 2
                    - (L - 3) * (2 ** (j - 1) - S[j - 1][n] - 1 / 2) / 3
                    + 1 / 2
                )
            nuj.append(nujn)
        nu.append(nuj)
    return nu


def get_gamma(name, J, N):
    """
    Compute the indices of the last boundary coefficient on the left-side
    and of the last boundary coefficient on the right-side (that is the
    coefficients that are affected by circularity)

    Input:
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Maximum level for DWPT
        type N = integer
        N = Length of the time series
    Output:
        type gamB = list of J values
        gamB = Each element of the list is made of the list of gammaj,n
               with n = 0, ... , 2 ** (j - 1) corresponding to the first
               left-side boundary coefficients
        type gamE = list of J values
        gamE = Each element of the list is made of the list of gammaj,n
               with n = 0, ... , 2 ** (j - 1) corresponding to the last
               left-side boundary coefficients
    """
    nu = get_nu(name, J)
    if name[0:2] == 'LA':
        L = int(name[2:])
    else:
        L = int(name[1:])
    gamB = []
    gamE = []
    for j in range(1, J + 1):
        gamBj = []
        gamEj = []
        t = int(floor((L - 2) * (1 - 1.0 / (2**j))))
        for n in range(0, 2**j):
            gamBj.append(int((2**j * (t + 1) - 1 - abs(nu[j - 1][n])) % N))
        t = 0
        for n in range(0, 2**j):
            gamEj.append(int((2**j * (t + 1) - 1 - abs(nu[j - 1][n])) % N))
        gamB.append(gamBj)
        gamE.append(gamEj)
    return (gamB, gamE)
