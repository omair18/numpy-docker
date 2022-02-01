#!/usr/bin/env python
# coding: utf-8

import numpy as np 

if __name__ == "__main__":
    A = np.array([1,2,3])
    B = np.array([3,4,5])
    C = A*B
    print("C.shape = ", C.shape)
    print("C = ", C)