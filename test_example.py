import numpy as np 
import os

class dummyTracker():
    def __init__(self):
        self.trackData = [[1,10,15,20,25]]
    def get(self):
        return self.trackData[0]

def incr(x):
    return x + 1

def test_incr():
    zeros = np.zeros((1,3),dtype=np.float32)
    print("HELLLLLLOOOOOO", zeros.shape)
    #os.system('python3 basic_classification.py')
    #assert incr(3) == 5
    assert zeros.shape[1] == 3

def test_tracker():
    groundtruth = [1,10,15,20,23]
    t = dummyTracker()
    tracks = t.get()
    t1 = np.array(groundtruth[1:]) #only getting the coordinates 
    t2 = np.array(tracks[1:])
    distance = np.linalg.norm(t1-t2)
    print("Distance computed = ", distance)
    assert distance < 3.0


