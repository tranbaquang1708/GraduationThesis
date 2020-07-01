import sys

import polyscope as ps
import numpy as np

argvs = sys.argv
if len(argvs)!=2:
    print('Usage: python main.py inputpc.xyz')
    sys.exit()

filename = argvs[1]

pc = np.loadtxt(filename)
# pc[:,0:3] contains the points coordinates
# pc[:,3:6] contains the surface normals at each point

ps.init()
ps.register_point_cloud("my pc", pc[:,0:3])
ps.show()
