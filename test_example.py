import numpy as np 
import os
def incr(x):
    return x + 1

def test_incr():
    zeros = np.zeros((1,3),dtype=np.float32)
    print("HELLLLLLOOOOOO", zeros.shape)
    #assert incr(3) == 5
    assert zeros.shape[1] == 2
