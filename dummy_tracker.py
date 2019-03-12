import numpy as np
#from scipy.spatial import distance as sci_distance

class dummyTracker():
    def __init__(self):
        self.trackData = [[1,10,15,20,25]]
    def get(self):
        return self.trackData[0]

groundtruth = [1,10,15,20,23]
t = dummyTracker()
tracks = t.get()
t1 = np.array(groundtruth[1:]) #only getting the coordinates 
t2 = np.array(tracks[1:])
distance = np.linalg.norm(t1-t2)
print(distance)
