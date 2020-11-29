import numpy as np 
from scipy.optimize import leastsq
import pylab as pl
x = ([2,4,8,16])

block_linear_y = ([7.464915,4.94,2.655,1.928,1.28])

block_tree_y = ([7.495866,5.100265,2.447935,1.430920])

pl.plot(x, block_tree_y, 'b^-', label='Origin Line')
#pl.plot(x, p1(x), 'gv--', label='Poly Fitting Line(deg=3)')
pl.show()