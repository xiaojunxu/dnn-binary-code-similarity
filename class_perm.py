NUM = 7305
import numpy as np
perm = np.random.permutation(NUM)
ofile = np.save("class_perm", perm)
