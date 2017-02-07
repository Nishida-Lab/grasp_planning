import pcl
import shutil
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__' :
    shutil.copy("../../grasp_dataset/01/pcd0100.txt","pcd0100.pcd")
    cloud = pcl.load("pcd0100.pcd")
    os.remove("pcd0100.pcd")
    a = np.asarray(cloud)

    x = []
    y = []
    z = []
    for i in range(0,len(a),50):
        x.append(a[i][0])
        y.append(a[i][1])
        z.append(a[i][2])

    i = 1
    fig1 = plt.figure(i)
    ax = Axes3D(fig1)
    p1 = ax.scatter3D(x,y,z,color=(1.0,0,0),marker='o',s=1)

    plt.show()
