import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

def test(int e):
    cdef np.ndarray X = np.array([
        [-2, 4, -1],
        [4, 1, -1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],

    ], dtype=DTYPE)

    cdef np.ndarray y = np.array([-1, -1, 1, 1, 1], dtype=DTYPE)

    cdef np.ndarray w = np.zeros(len(X[0]), dtype=DTYPE)

    cdef int epochs = e
    cdef int eta = 1
    cdef int m = len(X)

    for epoch in range(1, epochs):
        for i in range(0,m):
            if (y[i] * np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * y[i]) + (-2 * (1 / epoch) * w))
            else:
                w = w + eta * (-2 * (1 / epoch) * w)
