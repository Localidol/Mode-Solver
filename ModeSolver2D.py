import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import eigs, eigsh
from math import pi


# Variables (nm)
wl = 1310
c = 3e8
frq = c/wl
mu0 = 4*pi*1e-7
eps0 = 8.8541e-12
k0 = 2*pi*frq*np.sqrt(mu0*eps0)
eps_Si = 12.25
eps_o2 = 2.10
width = 400
dx = 5
dy = dx
h = 220
slab = 0
oxide = 500 # Height
Lslab = 1000 # For each side
boundary = '0000'

guess = np.sqrt(eps_Si)

# Top waveguide
wg = eps_Si*np.ones([int(width/dx), int(h/dx)])
wg = np.concatenate((wg, eps_o2*np.ones([int(Lslab/dx), int(h/dx)])), axis=0)
wg = np.concatenate((eps_o2*np.ones([int(Lslab/dx), int(h/dx)]), wg), axis=0)
# Slab
a= wg.shape
wg = np.concatenate((wg, eps_Si*np.ones([a[0], int(slab/dx)])), axis=1)
# Oxide below and on top
wg = np.concatenate((wg, eps_o2*np.ones([a[0], int(oxide/dx)])), axis=1)
wg = np.concatenate((eps_o2*np.ones([a[0], int(oxide/dx)]), wg), axis=1)

plt.figure(8)
plt.imshow(wg.T)
plt.colorbar(orientation='horizontal')
# plt.show()

eps = wg
del wg


guess = 3.5
nmodes = 1


nx, ny = eps.shape
print(nx, ny)

# Valid only with isotropic material
epsxx = eps
epsyy = epsxx
epsxy = np.zeros(eps.shape)
epsyx = np.zeros(eps.shape)
epszz = epsxx

nx += 1
ny += 1

# Padding
epsxx = np.column_stack((epsxx[:, 0], epsxx, epsxx[:, -1]))
epsxx = np.row_stack((epsxx[0, :], epsxx, epsxx[-1, :]))

epsyy = np.column_stack((epsyy[:, 0], epsyy, epsyy[:, -1]))
epsyy = np.row_stack((epsyy[0, :], epsyy, epsyy[-1, :]))

epsxy = np.column_stack((epsxy[:, 0], epsxy, epsxy[:, -1]))
epsxy = np.row_stack((epsxy[0, :], epsxy, epsxy[-1, :]))

epsyx = np.column_stack((epsyx[:, 0], epsyx, epsyx[:, -1]))
epsyx = np.row_stack((epsyx[0, :], epsyx, epsyx[-1, :]))

epszz = np.column_stack((epszz[:, 0], epszz, epszz[:, -1]))
epszz = np.row_stack((epszz[0, :], epszz, epszz[-1, :]))


# Dimensions should be ok.

# Free-space wavevector
k = 2*pi/wl

dx = dx*np.ones([nx+1, 1])
dy = dy*np.ones([1, ny+1])

# Distance to neighboring points to north south east and west
# relative to the point under investigation (P)

n = (np.ones([nx, 1])*dy[0, 1:]).reshape(1, nx*ny, order='F')
s = (np.ones([nx, 1])*dy[0, 0:-1]).reshape(1, nx*ny, order='F')
e = (dx[1:]*np.ones([ny])).reshape(1, nx*ny, order='F') 
w = (dx[0:-1]*np.ones([ny])).reshape(1, nx*ny, order='F')



# Fine????

# Creation of the tensor?? Understand and describe better

exx1 = epsxx[:-1, 1:].reshape(1, nx*ny, order='F')
exx2 = epsxx[:-1, :-1].reshape(1, nx*ny, order='F')
exx3 = epsxx[1:, :-1].reshape(1, nx*ny, order='F')
exx4 = epsxx[1:, 1:].reshape(1, nx*ny, order='F')

eyy1 = epsyy[:-1, 1:].reshape(1, nx*ny, order='F')
eyy2 = epsyy[:-1, :-1].reshape(1, nx*ny, order='F')
eyy3 = epsyy[1:, :-1].reshape(1, nx*ny, order='F')
eyy4 = epsyy[1:, 1:].reshape(1, nx*ny, order='F')

exy1 = epsxy[:-1, 1:].reshape(1, nx*ny, order='F')
exy2 = epsxy[:-1, :-1].reshape(1, nx*ny, order='F')
exy3 = epsxy[1:, :-1].reshape(1, nx*ny, order='F')
exy4 = epsxy[1:, 1:].reshape(1, nx*ny, order='F')

eyx1 = epsyx[:-1, 1:].reshape(1, nx*ny, order='F')
eyx2 = epsyx[:-1, :-1].reshape(1, nx*ny, order='F')
eyx3 = epsyx[1:, :-1].reshape(1, nx*ny, order='F')
eyx4 = epsyx[1:, 1:].reshape(1, nx*ny, order='F')

ezz1 = epszz[:-1, 1:].reshape(1, nx*ny, order='F')
ezz2 = epszz[:-1, :-1].reshape(1, nx*ny, order='F')
ezz3 = epszz[1:, :-1].reshape(1, nx*ny, order='F')
ezz4 = epszz[1:, 1:].reshape(1, nx*ny, order='F')


ns21 = n*eyy2 + s*eyy1
ns34 = n*eyy3 + s*eyy4

ew14 = e*exx1 + w*exx4
ew23 = e*exx2 + w*exx3

axxn = ((2*eyy4*e-eyx4*n)*(eyy3/ezz4)/ns34 + (2*eyy1*w+eyx1*n)*(eyy2/ezz1)/ns21)/(n*(e+w))
axxs = ((2*eyy3*e+eyx3*s)*(eyy4/ezz3)/ns34 + (2*eyy2*w-eyx2*s)*(eyy1/ezz2)/ns21)/(s*(e+w))
ayye = (2*n*exx4 - e*exy4)*exx1/ezz4/e/ew14/(n+s) + (2*s*exx3 + e*exy3)*exx2/ezz3/e/ew23/(n+s)
ayyw = (2*exx1*n + exy1*w)*exx4/ezz1/w/ew14/(n+s) + (2*exx2*s - exy2*w)*exx3/ezz2/w/ew23/(n+s)


axxe = 2/(e*(e+w)) + (eyy4*eyx3/ezz3 - eyy3*eyx4/ezz4)/(e+w)/ns34

axxw = 2/(w*(e+w)) + (eyy2*eyx1/ezz1 - eyy1*eyx2/ezz2)/(e+w)/ns21

ayyn = 2/(n*(n+s)) + (exx4*exy1/ezz1 - exx1*exy4/ezz4)/(n+s)/ew14

ayys = 2/(s*(n+s)) + (exx2*exy3/ezz3 - exx3*exy2/ezz2)/(n+s)/ew23

axxne = +eyx4*eyy3/ezz4/(e+w)/ns34
axxse = -eyx3*eyy4/ezz3/(e+w)/ns34
axxnw = -eyx1*eyy2/ezz1/(e+w)/ns21
axxsw = +eyx2*eyy1/ezz2/(e+w)/ns21

ayyne = +exy4*exx1/ezz4/(n+s)/ew14
ayyse = -exy3*exx2/ezz3/(n+s)/ew23
ayynw = -exy1*exx4/ezz1/(n+s)/ew14
ayysw = +exy2*exx3/ezz2/(n+s)/ew23

axxp = - axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw + (k**2)*(n+s)*(eyy4*eyy3*e/ns34 + eyy1*eyy2*w/ns21)/(e+w)

ayyp = - ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw + (k**2)*(e+w)*(exx1*exx4*n/ew14 + exx2*exx3*s/ew23)/(n+s)

axyn = (eyy3*eyy4/ezz4/ns34 - eyy2*eyy1/ezz1/ns21 + s*(eyy2*eyy4 - eyy1*eyy3)/ns21/ns34)/(e+w)

axys = (eyy1*eyy2/ezz2/ns21 - eyy4*eyy3/ezz3/ns34 + n*(eyy2*eyy4 - eyy1*eyy3)/ns21/ns34)/(e+w)

ayxe = (exx1*exx4/ezz4/ew14 - exx2*exx3/ezz3/ew23 + w*(exx2*exx4 - exx1*exx3)/ew23/ew14)/(n+s)

ayxw = (exx3*exx2/ezz2/ew23 - exx4*exx1/ezz1/ew14 + e*(exx4*exx2 - exx1*exx3)/ew23/ew14)/(n+s)


axye = (eyy4*(1-eyy3/ezz3) - eyy3*(1-eyy4/ezz4))/ns34/(e+w) - \
       2*(eyx1*eyy2/ezz1*n*w/ns21 + \
          eyx2*eyy1/ezz2*s*w/ns21 + \
          eyx4*eyy3/ezz4*n*e/ns34 + \
          eyx3*eyy4/ezz3*s*e/ns34 + \
          eyy1*eyy2*(1/ezz1-1/ezz2)*(w**2)/ns21 + \
          eyy3*eyy4*(1/ezz4-1/ezz3)*e*w/ns34)/e/(e+w)**2

axyw = (eyy2*(1-eyy1/ezz1) - eyy1*(1-eyy2/ezz2))/ns21/(e+w) - \
       2*(eyx4*eyy3/ezz4*n*e/ns34 + \
          eyx3*eyy4/ezz3*s*e/ns34 + \
          eyx1*eyy2/ezz1*n*w/ns21 + \
          eyx2*eyy1/ezz2*s*w/ns21 + \
          eyy4*eyy3*(1/ezz3-1/ezz4)*(e**2)/ns34 + \
          eyy2*eyy1*(1/ezz2-1/ezz1)*w*e/ns21)/w/(e+w)**2 

ayxn = (exx4*(1-exx1/ezz1) - exx1*(1-exx4/ezz4))/ew14/(n+s) - \
       2*(exy3*exx2/ezz3*e*s/ew23 + \
          exy2*exx3/ezz2*w*s/ew23 + \
          exy4*exx1/ezz4*e*n/ew14 + \
          exy1*exx4/ezz1*w*n/ew14 + \
          exx3*exx2*(1/ezz3-1/ezz2)*(s**2)/ew23 + \
          exx1*exx4*(1/ezz4-1/ezz1)*n*s/ew14)/n/(n+s)**2

ayxs = (exx2*(1-exx3/ezz3) - exx3*(1-exx2/ezz2))/ew23/(n+s) - \
       2*(exy4*exx1/ezz4*e*n/ew14 + \
          exy1*exx4/ezz1*w*n/ew14 + \
          exy3*exx2/ezz3*e*s/ew23 + \
          exy2*exx3/ezz2*w*s/ew23 + \
          exx4*exx1*(1/ezz1-1/ezz4)*(n**2)/ew14 + \
          exx2*exx3*(1/ezz2-1/ezz3)*s*n/ew23)/s/(n+s)**2

axyne = +eyy3*(1-eyy4/ezz4)/(e+w)/ns34
axyse = -eyy4*(1-eyy3/ezz3)/(e+w)/ns34
axynw = -eyy2*(1-eyy1/ezz1)/(e+w)/ns21
axysw = +eyy1*(1-eyy2/ezz2)/(e+w)/ns21

ayxne = +exx1*(1-exx4/ezz4)/(n+s)/ew14
ayxse = -exx2*(1-exx3/ezz3)/(n+s)/ew23
ayxnw = -exx4*(1-exx1/ezz1)/(n+s)/ew14
ayxsw = +exx3*(1-exx2/ezz2)/(n+s)/ew23

axyp = -(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw) \
       - (k**2)*(w*(n*eyx1*eyy2 + s*eyx2*eyy1)/ns21 + \
        e*(s*eyx3*eyy4 + n*eyx4*eyy3)/ns34)/(e+w)

ayxp = -(ayxn + ayxs + ayxe + ayxw + ayxne + ayxse + ayxnw + ayxsw) \
       - (k**2)*(n*(w*exy1*exx4 + e*exy4*exx1)/ew14 + \
        s*(w*exy2*exx3 + e*exy3*exx2)/ew23)/(n+s)



## Boundaries
ii = np.linspace(0, nx*ny-1, nx*ny, dtype=int)
ii = ii.reshape((nx, ny), order='F')

#North boundary
ib = ii[:, -1]

if boundary[0] == '0':
    sign = 0
elif boundary[0] == 'S':
    sign = +1
elif boundary[0] == 'A':
    sign = -1
else:
    print('The north boundary condition is not recognized!')    


axxs[0, ib] = axxs[0, ib] + sign*axxn[0, ib]
axxse[0, ib] = axxse[0, ib] + sign*axxne[0, ib]
axxsw[0, ib] = axxsw[0, ib] + sign*axxnw[0, ib]
ayxs[0, ib] = ayxs[0, ib] + sign*ayxn[0, ib]
ayxse[0, ib] = ayxse[0, ib] + sign*ayxne[0, ib]
ayxsw[0, ib] = ayxsw[0, ib] + sign*ayxnw[0, ib]
ayys[0, ib] = ayys[0, ib] - sign*ayyn[0, ib]
ayyse[0, ib] = ayyse[0, ib] - sign*ayyne[0, ib]
ayysw[0, ib] = ayysw[0, ib] - sign*ayynw[0, ib]
axys[0, ib] = axys[0, ib] - sign*axyn[0, ib]
axyse[0, ib] = axyse[0, ib] - sign*axyne[0, ib]
axysw[0, ib] = axysw[0, ib] - sign*axynw[0, ib]


# South boundary
ib = ii[:, 0]

if boundary[1] == '0':
    sign = 0
elif boundary[1] == 'S':
    sign = +1
elif boundary[1] == 'A':
    sign = -1
else:
    print('The south boundary condition is not recognized!')    

axxn[0, ib]  = axxn[0, ib]  + sign*axxs[0, ib]
axxne[0, ib] = axxne[0, ib] + sign*axxse[0, ib]
axxnw[0, ib] = axxnw[0, ib] + sign*axxsw[0, ib]
ayxn[0, ib]  = ayxn[0, ib]  + sign*ayxs[0, ib]
ayxne[0, ib] = ayxne[0, ib] + sign*ayxse[0, ib]
ayxnw[0, ib] = ayxnw[0, ib] + sign*ayxsw[0, ib]
ayyn[0, ib]  = ayyn[0, ib]  - sign*ayys[0, ib]
ayyne[0, ib] = ayyne[0, ib] - sign*ayyse[0, ib]
ayynw[0, ib] = ayynw[0, ib] - sign*ayysw[0, ib]
axyn[0, ib]  = axyn[0, ib]  - sign*axys[0, ib]
axyne[0, ib] = axyne[0, ib] - sign*axyse[0, ib]
axynw[0, ib] = axynw[0, ib] - sign*axysw[0, ib]


# East boundary
ib = ii[-1,:]

if boundary[2] == '0':
    sign = 0
elif boundary[2] == 'S':
    sign = +1
elif boundary[2] == 'A':
    sign = -1
else:
    print('The south boundary condition is not recognized!')


axxw[0, ib]  = axxw[0, ib]  + sign*axxe[0, ib]
axxnw[0, ib] = axxnw[0, ib] + sign*axxne[0, ib]
axxsw[0, ib] = axxsw[0, ib] + sign*axxse[0, ib]
ayxw[0, ib]  = ayxw[0, ib]  + sign*ayxe[0, ib]
ayxnw[0, ib] = ayxnw[0, ib] + sign*ayxne[0, ib]
ayxsw[0, ib] = ayxsw[0, ib] + sign*ayxse[0, ib]
ayyw[0, ib]  = ayyw[0, ib]  - sign*ayye[0, ib]
ayynw[0, ib] = ayynw[0, ib] - sign*ayyne[0, ib]
ayysw[0, ib] = ayysw[0, ib] - sign*ayyse[0, ib]
axyw[0, ib]  = axyw[0, ib]  - sign*axye[0, ib]
axynw[0, ib] = axynw[0, ib] - sign*axyne[0, ib]
axysw[0, ib] = axysw[0, ib] - sign*axyse[0, ib]


# West boundary
ib = ii[0, :]

if boundary[3] == '0':
    sign = 0
elif boundary[3] == 'S':
    sign = +1
elif boundary[3] == 'A':
    sign = -1
else:
    print('The south boundary condition is not recognized!')


axxe[0, ib] = axxe[0, ib] + sign * axxw[0, ib]
axxne[0, ib] = axxne[0, ib] + sign * axxnw[0, ib]
axxse[0, ib] = axxse[0, ib] + sign * axxsw[0, ib]
ayxe[0, ib] = ayxe[0, ib] + sign * ayxw[0, ib]
ayxne[0, ib] = ayxne[0, ib] + sign * ayxnw[0, ib]
ayxse[0, ib] = ayxse[0, ib] + sign * ayxsw[0, ib]
ayye[0, ib] = ayye[0, ib] - sign * ayyw[0, ib]
ayyne[0, ib] = ayyne[0, ib] - sign * ayynw[0, ib]
ayyse[0, ib] = ayyse[0, ib] - sign * ayysw[0, ib]
axye[0, ib] = axye[0, ib] - sign * axyw[0, ib]
axyne[0, ib] = axyne[0, ib] - sign * axynw[0, ib]
axyse[0, ib] = axyse[0, ib] - sign * axysw[0, ib]


## Creating the sparse matrices
# Indices
iall = ii.reshape(1, nx*ny, order='F')
iss = ii[:, :-1].reshape(1, nx*(ny-1), order='F')
inn = ii[:, 1:].reshape(1, nx*(ny-1), order='F')
ie = ii[1:, :].reshape(1, (nx-1)*ny, order='F')
iw = ii[:-1, :].reshape(1, (nx-1)*ny, order='F')
ine = ii[1:, 1:].reshape(1, (nx-1)*(ny-1), order='F')
ise = ii[1:, :-1].reshape(1, (nx-1)*(ny-1), order='F')
isw = ii[:-1, :-1].reshape(1, (nx-1)*(ny-1), order='F')
inw = ii[:-1, 1:].reshape(1, (nx-1)*(ny-1), order='F')

# Axx
Axx = coo_matrix((axxp[0, iall[0, :]], (iall[0, :], iall[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxe[0, iw[0, :]], (iw[0, :], ie[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxw[0, ie[0, :]], (ie[0, :], iw[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxn[0, iss[0, :]], (iss[0, :], inn[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxs[0, inn[0, :]], (inn[0, :], iss[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxsw[0, ine[0, :]], (ine[0, :], isw[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxnw[0, ise[0, :]], (ise[0, :], inw[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxne[0, isw[0, :]], (isw[0, :], ine[0, :])), shape=(nx*ny, nx*ny))
Axx += coo_matrix((axxse[0, inw[0, :]], (inw[0, :], ise[0, :])), shape=(nx*ny, nx*ny))
Axx = Axx.tocsr()
#np.savetxt('Axx_py.txt', Axx.toarray())

# Axy
Axy = coo_matrix((axyp[0, iall[0, :]], (iall[0, :], iall[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axye[0, iw[0, :]], (iw[0, :], ie[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axyw[0, ie[0, :]], (ie[0, :], iw[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axyn[0, iss[0, :]], (iss[0, :], inn[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axys[0, inn[0, :]], (inn[0, :], iss[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axysw[0, ine[0, :]], (ine[0, :], isw[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axynw[0, ise[0, :]], (ise[0, :], inw[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axyne[0, isw[0, :]], (isw[0, :], ine[0, :])), shape=(nx*ny, nx*ny))
Axy += coo_matrix((axyse[0, inw[0, :]], (inw[0, :], ise[0, :])), shape=(nx*ny, nx*ny))
Axy = Axy.tocsr()
#np.savetxt('Axy_py.txt', Axy.toarray())

# Ayx
Ayx = coo_matrix((ayxp[0, iall[0, :]], (iall[0, :], iall[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxe[0, iw[0, :]], (iw[0, :], ie[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxw[0, ie[0, :]], (ie[0, :], iw[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxn[0, iss[0, :]], (iss[0, :], inn[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxs[0, inn[0, :]], (inn[0, :], iss[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxsw[0, ine[0, :]], (ine[0, :], isw[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxnw[0, ise[0, :]], (ise[0, :], inw[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxne[0, isw[0, :]], (isw[0, :], ine[0, :])), shape=(nx*ny, nx*ny))
Ayx += coo_matrix((ayxse[0, inw[0, :]], (inw[0, :], ise[0, :])), shape=(nx*ny, nx*ny))
Ayx = Ayx.tocsr()
#np.savetxt('Ayx_py.txt', Ayx.toarray())

# Ayy
Ayy = coo_matrix((ayyp[0, iall[0, :]], (iall[0, :], iall[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayye[0, iw[0, :]], (iw[0, :], ie[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayyw[0, ie[0, :]], (ie[0, :], iw[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayyn[0, iss[0, :]], (iss[0, :], inn[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayys[0, inn[0, :]], (inn[0, :], iss[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayysw[0, ine[0, :]], (ine[0, :], isw[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayynw[0, ise[0, :]], (ise[0, :], inw[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayyne[0, isw[0, :]], (isw[0, :], ine[0, :])), shape=(nx*ny, nx*ny))
Ayy += coo_matrix((ayyse[0, inw[0, :]], (inw[0, :], ise[0, :])), shape=(nx*ny, nx*ny))
Ayy = Ayy.tocsr()
#np.savetxt('Ayy_py.txt', Ayy.toarray())




At = hstack([Axx, Axy])
Ab = hstack([Ayx, Ayy])

A = vstack([At, Ab]).tocsr()

del At, Ab

#[t1, t2] = A.shape

print('Starting!')
shift = (guess*k)**2
[v, d] = eigs(A, nmodes, sigma=shift, tol=1e-8)
#[v, d] = eigs(A, 1, M=sp.eye(t1, t2, format='csr'), sigma=guess, tol=1e-8)

neff = wl*np.sqrt(np.diag(v))/(2*pi)
print(neff)
#print('Neff: %.5f' %np.asscalar(neff.real))

phix = np.zeros([nx, ny, nmodes])
phiy = np.zeros([nx, ny, nmodes])
temp = np.zeros([nx*ny, 2], dtype='complex128')

for kk in range(nmodes):
    temp[:, 0] = d[:nx*ny, 0]
    temp[:, 1] = d[nx*ny:, 0]
    mag = np.max(np.sqrt(np.sum(np.abs(temp)**2, axis=1)))
    ii = np.argmax(np.sqrt(np.sum(np.abs(temp)**2, axis=1)))
    
    if np.abs(temp[ii, 0]) > np.abs(temp[ii, 1]):
        jj = 0
    else:
        jj = 1
    
    mag = mag*temp[ii,jj]/abs(temp[ii,jj])
    temp = temp/mag
    
    phix[:, :, kk] = temp[:, 0].reshape(nx, ny, order='F')
    phiy[:, :, kk] = temp[:, 1].reshape(nx, ny, order='F')
    
#    phix(:,:,kk) = reshape(temp(:,1),nx,ny);
#    phiy(:,:,kk) = reshape(temp(:,2),nx,ny);

plt.figure(1)
plt.title('Hx')
plt.imshow(np.real(phix[:, :, 0]).T, interpolation='nearest')
plt.colorbar(orientation='horizontal')


plt.figure(2)
plt.title('Hy')
plt.imshow(np.real(phiy[:, :, 0]).T, interpolation='nearest')
plt.colorbar(orientation='horizontal')

# plt.show()

print('Yoooo!!!')


nx, ny = eps.shape

# Valid only with isotropic material
epsxx = eps
epsyy = epsxx
epsxy = np.zeros(eps.shape)
epsyx = np.zeros(eps.shape)
epszz = epsxx

nx += 1
ny += 1

# Padding
epsxx = np.column_stack((epsxx[:, 0], epsxx, epsxx[:, -1]))
epsxx = np.row_stack((epsxx[0, :], epsxx, epsxx[-1, :]))

epsyy = np.column_stack((epsyy[:, 0], epsyy, epsyy[:, -1]))
epsyy = np.row_stack((epsyy[0, :], epsyy, epsyy[-1, :]))

epsxy = np.column_stack((epsxy[:, 0], epsxy, epsxy[:, -1]))
epsxy = np.row_stack((epsxy[0, :], epsxy, epsxy[-1, :]))

epsyx = np.column_stack((epsyx[:, 0], epsyx, epsyx[:, -1]))
epsyx = np.row_stack((epsyx[0, :], epsyx, epsyx[-1, :]))

epszz = np.column_stack((epszz[:, 0], epszz, epszz[:, -1]))
epszz = np.row_stack((epszz[0, :], epszz, epszz[-1, :]))


# Dimensions should be ok.

# Free-space wavevector
k = 2*pi/wl

# Propagation constant (eigenvalue)
b = neff*k       

# Only valid for uniform grid
dx = dx*np.ones([nx+1, 1])
dy = dy*np.ones([1, ny+1])

# Distance to neighboring points to north south east and west
# relative to the point under investigation (P)
#
# epsilon tensor elements in regions 1,2,3,4, relative to the
# mesh point under consideration (P), as shown below.
#
#                 NW------N------NE
#                 |       |       |
#                 |   1   n   4   |
#                 |       |       |
#                 W---w---P---e---E
#                 |       |       |
#                 |   2   s   3   |
#                 |       |       |
#                 SW------S------SE


n = (np.ones([nx, 1])*dy[0, 1:]).reshape(1, nx*ny, order='F')
s = (np.ones([nx, 1])*dy[0, 0:-1]).reshape(1, nx*ny, order='F')
e = (dx[1:]*np.ones([ny])).reshape(1, nx*ny, order='F') 
w = (dx[0:-1]*np.ones([ny])).reshape(1, nx*ny, order='F')


# Creation of the tensor?? Understand and describe better

exx1 = epsxx[:-1, 1:].reshape(1, nx*ny, order='F')
exx2 = epsxx[:-1, :-1].reshape(1, nx*ny, order='F')
exx3 = epsxx[1:, :-1].reshape(1, nx*ny, order='F')
exx4 = epsxx[1:, 1:].reshape(1, nx*ny, order='F')

eyy1 = epsyy[:-1, 1:].reshape(1, nx*ny, order='F')
eyy2 = epsyy[:-1, :-1].reshape(1, nx*ny, order='F')
eyy3 = epsyy[1:, :-1].reshape(1, nx*ny, order='F')
eyy4 = epsyy[1:, 1:].reshape(1, nx*ny, order='F')

exy1 = epsxy[:-1, 1:].reshape(1, nx*ny, order='F')
exy2 = epsxy[:-1, :-1].reshape(1, nx*ny, order='F')
exy3 = epsxy[1:, :-1].reshape(1, nx*ny, order='F')
exy4 = epsxy[1:, 1:].reshape(1, nx*ny, order='F')

eyx1 = epsyx[:-1, 1:].reshape(1, nx*ny, order='F')
eyx2 = epsyx[:-1, :-1].reshape(1, nx*ny, order='F')
eyx3 = epsyx[1:, :-1].reshape(1, nx*ny, order='F')
eyx4 = epsyx[1:, 1:].reshape(1, nx*ny, order='F')

ezz1 = epszz[:-1, 1:].reshape(1, nx*ny, order='F')
ezz2 = epszz[:-1, :-1].reshape(1, nx*ny, order='F')
ezz3 = epszz[1:, :-1].reshape(1, nx*ny, order='F')
ezz4 = epszz[1:, 1:].reshape(1, nx*ny, order='F')


bzxne = (1/2*(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*eyx4/ezz4/ \
         (n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/ \
         (e+w)*eyy3*eyy1*w*eyy2+1/2*(ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)* \
         (1-exx4/ezz4)/ezz3/ezz2/(w*exx3+e*exx2)/ \
         (w*exx4+e*exx1)/(n+s)*exx2*exx3*exx1*s)/b

bzxse = (-1/2*(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*eyx3/ezz3/ \
         (n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/ \
         (e+w)*eyy4*eyy1*w*eyy2+1/2*(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)* \
         (1-exx3/ezz3)/(w*exx3+e*exx2)/ezz4/ezz1/ \
         (w*exx4+e*exx1)/(n+s)*exx2*n*exx1*exx4)/b

bzxnw = (-1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*eyx1/ezz4/ \
         ezz3/(n*eyy3+s*eyy4)/ezz1/(n*eyy2+s*eyy1)/ \
         (e+w)*eyy4*eyy3*eyy2*e-1/2*(ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)* \
         (1-exx1/ezz1)/ezz3/ezz2/(w*exx3+e*exx2)/ \
         (w*exx4+e*exx1)/(n+s)*exx2*exx3*exx4*s)/b

bzxsw = (1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*eyx2/ezz4/ezz3/ \
         (n*eyy3+s*eyy4)/ezz2/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3* \
         eyy1*e-1/2*(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*(1-exx2/ezz2)/ \
         (w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx3*n*exx1*exx4)/b

bzxn = ((1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*n*ezz1*ezz2/eyy1*(2*eyy1/ \
        ezz1/n**2+eyx1/ezz1/n/w)+1/2*(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)* \
        n*ezz4*ezz3/eyy4*(2*eyy4/ezz4/n**2-eyx4/ezz4/n/e))/ezz4/ezz3/ \
        (n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy1* \
        w*eyy2*e+((ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*(1/2*ezz4* \
        ((1-exx1/ezz1)/n/w-exy1/ezz1*(2/n**2-2/n**2*s/(n+s)))/exx1*ezz1* \
        w+(ezz4-ezz1)*s/n/(n+s)+1/2*ezz1*(-(1-exx4/ezz4)/n/e-exy4/ \
        ezz4*(2/n**2-2/n**2*s/(n+s)))/exx4*ezz4*e)-(ezz4/exx1*ezz1*w+ \
        ezz1/exx4*ezz4*e)*(-ezz3*exy2/n/(n+s)/exx2*w+(ezz3-ezz2)* \
        s/n/(n+s)-ezz2*exy3/n/(n+s)/exx3*e))/ezz3/ezz2/ \
        (w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/ \
        (n+s)*exx2*exx3*n*exx1*exx4*s)/b

bzxs = ((1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*s*ezz2*ezz1/eyy2* \
        (2*eyy2/ezz2/s**2-eyx2/ezz2/s/w)+1/2*(n*ezz1*ezz2/eyy1+s*ezz2* \
        ezz1/eyy2)*s*ezz3*ezz4/eyy3*(2*eyy3/ezz3/s**2+eyx3/ezz3/s/e))/ \
        ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4* \
        eyy3*eyy1*w*eyy2*e+((ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*(-ezz4* \
        exy1/s/(n+s)/exx1*w-(ezz4-ezz1)*n/s/(n+s)-ezz1*exy4/s/(n+s)/ \
        exx4*e)-(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*(1/2*ezz3* \
        (-(1-exx2/ezz2)/s/w-exy2/ezz2*(2/s**2-2/s**2*n/(n+s)))/exx2* \
        ezz2*w-(ezz3-ezz2)*n/s/(n+s)+1/2*ezz2*((1-exx3/ezz3)/s/e-exy3/ \
        ezz3*(2/s**2-2/s**2*n/(n+s)))/exx3*ezz3*e))/ezz3/ezz2/ \
        (w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)* \
        exx2*exx3*n*exx1*exx4*s)/b

bzxe = ((n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*(1/2*n*ezz4*ezz3/eyy4* \
        (2/e**2-eyx4/ezz4/n/e)+1/2*s*ezz3*ezz4/eyy3*(2/e**2+eyx3/ \
        ezz3/s/e))/ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/ \
        (e+w)*eyy4*eyy3*eyy1*w*eyy2*e+(-1/2*(ezz3/exx2*ezz2*w+ezz2/ \
        exx3*ezz3*e)*ezz1*(1-exx4/ezz4)/n/exx4*ezz4-1/2*(ezz4/exx1* \
        ezz1*w+ezz1/exx4*ezz4*e)*ezz2*(1-exx3/ezz3)/s/exx3*ezz3)/ \
        ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)* \
        exx2*exx3*n*exx1*exx4*s)/b

bzxw = ((-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*(1/2*n*ezz1*ezz2/eyy1* \
        (2/w**2+eyx1/ezz1/n/w)+1/2*s*ezz2*ezz1/eyy2*(2/w**2-eyx2/ \
        ezz2/s/w))/ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/ \
        (e+w)*eyy4*eyy3*eyy1*w*eyy2*e+(1/2*(ezz3/exx2*ezz2*w+ezz2/ \
        exx3*ezz3*e)*ezz4*(1-exx1/ezz1)/n/exx1*ezz1+1/2* \
        (ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*ezz3*(1-exx2/ezz2)/s/ \
        exx2*ezz2)/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/ \
        (w*exx4+e*exx1)/(n+s)*exx2*exx3*n*exx1*exx4*s)/b

bzxp = (((-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*(1/2*n*ezz1*ezz2/eyy1* \
        (-2/w**2-2*eyy1/ezz1/n**2+k**2*eyy1-eyx1/ezz1/n/w)+1/2*s*ezz2* \
        ezz1/eyy2*(-2/w**2-2*eyy2/ezz2/s**2+k**2*eyy2+eyx2/ezz2/s/w))+ \
        (n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*(1/2*n*ezz4*ezz3/eyy4* \
        (-2/e**2-2*eyy4/ezz4/n**2+k**2*eyy4+eyx4/ezz4/n/e)+1/2*s*ezz3* \
        ezz4/eyy3*(-2/e**2-2*eyy3/ezz3/s**2+k**2*eyy3-eyx3/ezz3/s/e)))/ \
        ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4* \
        eyy3*eyy1*w*eyy2*e+((ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)* \
        (1/2*ezz4*(-k**2*exy1-(1-exx1/ezz1)/n/w-exy1/ezz1*(-2/n**2-2/n**2* \
        (n-s)/s))/exx1*ezz1*w+(ezz4-ezz1)*(n-s)/n/s+1/2*ezz1* \
        (-k**2*exy4+(1-exx4/ezz4)/n/e-exy4/ezz4*(-2/n**2-2/n**2*(n-s)/s))/ \
        exx4*ezz4*e)-(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*(1/2*ezz3* \
        (-k**2*exy2+(1-exx2/ezz2)/s/w-exy2/ezz2*(-2/s**2+2/s**2*(n-s)/n))/ \
        exx2*ezz2*w+(ezz3-ezz2)*(n-s)/n/s+1/2*ezz2*(-k**2*exy3- \
        (1-exx3/ezz3)/s/e-exy3/ezz3*(-2/s**2+2/s**2*(n-s)/n))/ \
        exx3*ezz3*e))/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/ \
        (n+s)*exx2*exx3*n*exx1*exx4*s)/b


bzyne = (1/2*(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*(1-eyy4/ezz4)/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy3*eyy1*w*eyy2+1/2*(ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*exy4/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/(w*exx4+e*exx1)/(n+s)*exx2*exx3*exx1*s)/b;

bzyse = (-1/2*(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*(1-eyy3/ezz3)/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy1*w*eyy2+1/2*(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*exy3/ezz3/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx2*n*exx1*exx4)/b;

bzynw = (-1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*(1-eyy1/ezz1)/ezz4/ezz3/(n*eyy3+s*eyy4)/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy2*e-1/2*(ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*exy1/ezz3/ezz2/(w*exx3+e*exx2)/ezz1/(w*exx4+e*exx1)/(n+s)*exx2*exx3*exx4*s)/b;

bzysw = (1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*(1-eyy2/ezz2)/ezz4/ezz3/(n*eyy3+s*eyy4)/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy1*e-1/2*(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*exy2/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx3*n*exx1*exx4)/b;

bzyn = ((1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*ezz1*ezz2/eyy1*(1-eyy1/ezz1)/w-1/2*(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*ezz4*ezz3/eyy4*(1-eyy4/ezz4)/e)/ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy1*w*eyy2*e+(ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*(1/2*ezz4*(2/n**2+exy1/ezz1/n/w)/exx1*ezz1*w+1/2*ezz1*(2/n**2-exy4/ezz4/n/e)/exx4*ezz4*e)/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx2*exx3*n*exx1*exx4*s)/b;

bzys = ((-1/2*(-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*ezz2*ezz1/eyy2*(1-eyy2/ezz2)/w+1/2*(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*ezz3*ezz4/eyy3*(1-eyy3/ezz3)/e)/ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy1*w*eyy2*e-(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*(1/2*ezz3*(2/s**2-exy2/ezz2/s/w)/exx2*ezz2*w+1/2*ezz2*(2/s**2+exy3/ezz3/s/e)/exx3*ezz3*e)/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx2*exx3*n*exx1*exx4*s)/b;

bzye = (((-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*(-n*ezz2/eyy1*eyx1/e/(e+w)+(ezz1-ezz2)*w/e/(e+w)-s*ezz1/eyy2*eyx2/e/(e+w))+(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*(1/2*n*ezz4*ezz3/eyy4*(-(1-eyy4/ezz4)/n/e-eyx4/ezz4*(2/e**2-2/e**2*w/(e+w)))+1/2*s*ezz3*ezz4/eyy3*((1-eyy3/ezz3)/s/e-eyx3/ezz3*(2/e**2-2/e**2*w/(e+w)))+(ezz4-ezz3)*w/e/(e+w)))/ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy1*w*eyy2*e+(1/2*(ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*ezz1*(2*exx4/ezz4/e**2-exy4/ezz4/n/e)/exx4*ezz4*e-1/2*(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*ezz2*(2*exx3/ezz3/e**2+exy3/ezz3/s/e)/exx3*ezz3*e)/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx2*exx3*n*exx1*exx4*s)/b;

bzyw = (((-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*(1/2*n*ezz1*ezz2/eyy1*((1-eyy1/ezz1)/n/w-eyx1/ezz1*(2/w**2-2/w**2*e/(e+w)))-(ezz1-ezz2)*e/w/(e+w)+1/2*s*ezz2*ezz1/eyy2*(-(1-eyy2/ezz2)/s/w-eyx2/ezz2*(2/w**2-2/w**2*e/(e+w))))+(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*(-n*ezz3/eyy4*eyx4/w/(e+w)-s*ezz4/eyy3*eyx3/w/(e+w)-(ezz4-ezz3)*e/w/(e+w)))/ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy1*w*eyy2*e+(1/2*(ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*ezz4*(2*exx1/ezz1/w**2+exy1/ezz1/n/w)/exx1*ezz1*w-1/2*(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*ezz3*(2*exx2/ezz2/w**2-exy2/ezz2/s/w)/exx2*ezz2*w)/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx2*exx3*n*exx1*exx4*s)/b;

bzyp = (((-n*ezz4*ezz3/eyy4-s*ezz3*ezz4/eyy3)*(1/2*n*ezz1*ezz2/eyy1*(-k**2*eyx1-(1-eyy1/ezz1)/n/w-eyx1/ezz1*(-2/w**2+2/w**2*(e-w)/e))+(ezz1-ezz2)*(e-w)/e/w+1/2*s*ezz2*ezz1/eyy2*(-k**2*eyx2+(1-eyy2/ezz2)/s/w-eyx2/ezz2*(-2/w**2+2/w**2*(e-w)/e)))+(n*ezz1*ezz2/eyy1+s*ezz2*ezz1/eyy2)*(1/2*n*ezz4*ezz3/eyy4*(-k**2*eyx4+(1-eyy4/ezz4)/n/e-eyx4/ezz4*(-2/e**2-2/e**2*(e-w)/w))+1/2*s*ezz3*ezz4/eyy3*(-k**2*eyx3-(1-eyy3/ezz3)/s/e-eyx3/ezz3*(-2/e**2-2/e**2*(e-w)/w))+(ezz4-ezz3)*(e-w)/e/w))/ezz4/ezz3/(n*eyy3+s*eyy4)/ezz2/ezz1/(n*eyy2+s*eyy1)/(e+w)*eyy4*eyy3*eyy1*w*eyy2*e+((ezz3/exx2*ezz2*w+ezz2/exx3*ezz3*e)*(1/2*ezz4*(-2/n**2-2*exx1/ezz1/w**2+k**2*exx1-exy1/ezz1/n/w)/exx1*ezz1*w+1/2*ezz1*(-2/n**2-2*exx4/ezz4/e**2+k**2*exx4+exy4/ezz4/n/e)/exx4*ezz4*e)-(ezz4/exx1*ezz1*w+ezz1/exx4*ezz4*e)*(1/2*ezz3*(-2/s**2-2*exx2/ezz2/w**2+k**2*exx2+exy2/ezz2/s/w)/exx2*ezz2*w+1/2*ezz2*(-2/s**2-2*exx3/ezz3/e**2+k**2*exx3-exy3/ezz3/s/e)/exx3*ezz3*e))/ezz3/ezz2/(w*exx3+e*exx2)/ezz4/ezz1/(w*exx4+e*exx1)/(n+s)*exx2*exx3*n*exx1*exx4*s)/b;



## Boundaries
ii = np.linspace(0, nx*ny-1, nx*ny, dtype=int)
ii = ii.reshape((nx, ny), order='F')

#North boundary
ib = ii[:, -1]

if boundary[0] == '0':
    sign = 0
elif boundary[0] == 'S':
    sign = +1
elif boundary[0] == 'A':
    sign = -1
else:
    print('The north boundary condition is not recognized!')    

bzxs[0, ib]  = bzxs[0, ib]  + sign*bzxn[0, ib]
bzxse[0, ib] = bzxse[0, ib] + sign*bzxne[0, ib]
bzxsw[0, ib] = bzxsw[0, ib] + sign*bzxnw[0, ib]
bzys[0, ib]  = bzys[0, ib]  - sign*bzyn[0, ib]
bzyse[0, ib] = bzyse[0, ib] - sign*bzyne[0, ib]
bzysw[0, ib] = bzysw[0, ib] - sign*bzynw[0, ib]


# South boundary
ib = ii[:, 0]

if boundary[1] == '0':
   sign = 0
elif boundary[1] == 'S':
   sign = +1
elif boundary[1] == 'A':
   sign = -1
else:
   print('The south boundary condition is not recognized!')

bzxn[0, ib]  = bzxn[0, ib]  + sign*bzxs[0, ib]
bzxne[0, ib] = bzxne[0, ib] + sign*bzxse[0, ib]
bzxnw[0, ib] = bzxnw[0, ib] + sign*bzxsw[0, ib]
bzyn[0, ib]  = bzyn[0, ib]  - sign*bzys[0, ib]
bzyne[0, ib] = bzyne[0, ib] - sign*bzyse[0, ib]
bzynw[0, ib] = bzynw[0, ib] - sign*bzysw[0, ib]


# East boundary
ib = ii[-1,:]

if boundary[2] == '0':
   sign = 0
elif boundary[2] == 'S':
   sign = +1
elif boundary[2] == 'A':
   sign = -1
else:
   print('The south boundary condition is not recognized!')


bzxw[0, ib]  = bzxw[0, ib]  + sign*bzxe[0, ib]
bzxnw[0, ib] = bzxnw[0, ib] + sign*bzxne[0, ib]
bzxsw[0, ib] = bzxsw[0, ib] + sign*bzxse[0, ib]
bzyw[0, ib]  = bzyw[0, ib]  - sign*bzye[0, ib]
bzynw[0, ib] = bzynw[0, ib] - sign*bzyne[0, ib]
bzysw[0, ib] = bzysw[0, ib] - sign*bzyse[0, ib]


# West boundary
ib = ii[0, :]

if boundary[3] == '0':
   sign = 0
elif boundary[3] == 'S':
   sign = +1
elif boundary[3] == 'A':
   sign = -1
else:
   print('The south boundary condition is not recognized!')


bzxe[0, ib]  = bzxe[0, ib]  + sign*bzxw[0, ib]
bzxne[0, ib] = bzxne[0, ib] + sign*bzxnw[0, ib]
bzxse[0, ib] = bzxse[0, ib] + sign*bzxsw[0, ib]
bzye[0, ib]  = bzye[0, ib]  - sign*bzyw[0, ib]
bzyne[0, ib] = bzyne[0, ib] - sign*bzynw[0, ib]
bzyse[0, ib] = bzyse[0, ib] - sign*bzysw[0, ib]


## Creating the sparse matrices
# Indices
iall = ii.reshape(1, nx*ny, order='F')
iss = ii[:, :-1].reshape(1, nx*(ny-1), order='F')
inn = ii[:, 1:].reshape(1, nx*(ny-1), order='F')
ie = ii[1:, :].reshape(1, (nx-1)*ny, order='F')
iw = ii[:-1, :].reshape(1, (nx-1)*ny, order='F')
ine = ii[1:, 1:].reshape(1, (nx-1)*(ny-1), order='F')
ise = ii[1:, :-1].reshape(1, (nx-1)*(ny-1), order='F')
isw = ii[:-1, :-1].reshape(1, (nx-1)*(ny-1), order='F')
inw = ii[:-1, 1:].reshape(1, (nx-1)*(ny-1), order='F')


# Bzx
Bzx = coo_matrix((bzxp[0, iall[0, :]], (iall[0, :], iall[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxe[0, iw[0, :]], (iw[0, :], ie[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxw[0, ie[0, :]], (ie[0, :], iw[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxn[0, iss[0, :]], (iss[0, :], inn[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxs[0, inn[0, :]], (inn[0, :], iss[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxsw[0, ine[0, :]], (ine[0, :], isw[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxnw[0, ise[0, :]], (ise[0, :], inw[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxne[0, isw[0, :]], (isw[0, :], ine[0, :])), shape=(nx*ny, nx*ny))
Bzx += coo_matrix((bzxse[0, inw[0, :]], (inw[0, :], ise[0, :])), shape=(nx*ny, nx*ny))
Bzx = Bzx.tocsr()
#Bzx = Bzx.toarray()

# Bzy
Bzy = coo_matrix((bzyp[0, iall[0, :]], (iall[0, :], iall[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzye[0, iw[0, :]], (iw[0, :], ie[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzyw[0, ie[0, :]], (ie[0, :], iw[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzyn[0, iss[0, :]], (iss[0, :], inn[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzys[0, inn[0, :]], (inn[0, :], iss[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzysw[0, ine[0, :]], (ine[0, :], isw[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzynw[0, ise[0, :]], (ise[0, :], inw[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzyne[0, isw[0, :]], (isw[0, :], ine[0, :])), shape=(nx*ny, nx*ny))
Bzy += coo_matrix((bzyse[0, inw[0, :]], (inw[0, :], ise[0, :])), shape=(nx*ny, nx*ny))
Bzy = Bzy.tocsr()


B = hstack([Bzx, Bzy])
del Bzx, Bzy
print(B.shape)

#bbb = B.toarray()
#savemat('B.mat', {'B': bbb})

Hx = phix[:, :, 0]
Hy = phiy[:, :, 0]

Hz = B*np.hstack((Hx, Hy)).reshape(2*nx*ny, 1, order='F')
Hz = Hz.reshape(nx, ny, order='F')/1j
#savemat('Hz.mat', {'Hz': Hz})

# plt.figure(3)
# plt.imshow(np.real(Hz.T), interpolation='nearest')


nx -= 1
ny -= 1

# nx = nx-1;
# ny = ny-1;

exx = epsxx[1:-1, 1:-1]
exy = epsxy[1:-1, 1:-1]
eyx = epsyx[1:-1, 1:-1]
eyy = epsyy[1:-1, 1:-1]
ezz = epszz[1:-1, 1:-1]
edet = (exx*eyy - exy*eyx)

# exx = epsxx(2:nx+1,2:ny+1);
# exy = epsxy(2:nx+1,2:ny+1);
# eyx = epsyx(2:nx+1,2:ny+1);
# eyy = epsyy(2:nx+1,2:ny+1);
# ezz = epszz(2:nx+1,2:ny+1);
# edet = (exx.*eyy - exy.*eyx);

h = (dx[1:-1]*np.ones([ny])).reshape(1, nx*ny, order='F')
v = (np.ones([nx, 1])*dy[0, 1:-1]).reshape(1, nx*ny, order='F')

# h = dx(2:nx+1)*ones(1,ny);
# v = ones(nx,1)*dy(2:ny+1);

i1 = ii[:-1, 1:].reshape(1, nx*ny, order='F')
i2 = ii[:-1, :-1].reshape(1, nx*ny, order='F')
i3 = ii[1:, :-1].reshape(1, nx*ny, order='F')
i4 = ii[1:, 1:].reshape(1, nx*ny, order='F')

# i1 = ii(1:nx,2:ny+1);
# i2 = ii(1:nx,1:ny);
# i3 = ii(2:nx+1,1:ny);
# i4 = ii(2:nx+1,2:ny+1);

Hx = Hx.reshape(1, (nx+1)*(ny+1), order='F')
Hy = Hy.reshape(1, (nx+1)*(ny+1), order='F')
Hz = Hz.reshape(1, (nx+1)*(ny+1), order='F')


Dx = +neff*(Hy[0, i1] + Hy[0, i2] + Hy[0, i3] + Hy[0, i4])/4 + (Hz[0, i1] + Hz[0, i4] - Hz[0, i2] - Hz[0, i3])/(1j*2*k*v)
Dy = -neff*(Hx[0, i1] + Hx[0, i2] + Hx[0, i3] + Hx[0, i4])/4 - (Hz[0, i3] + Hz[0, i4] - Hz[0, i1] - Hz[0, i2])/(1j*2*k*h)
Dz = ((Hy[0, i3] + Hy[0, i4] - Hy[0, i1] - Hy[0, i2])/(2*h) - (Hx[0, i1] + Hx[0, i4] - Hx[0, i2] - Hx[0, i3])/(2*v))/(1j*k)

# Dx = +neff*(Hy(i1) + Hy(i2) + Hy(i3) + Hy(i4))/4 + ...
#      (Hz(i1) + Hz(i4) - Hz(i2) - Hz(i3))./(j*2*k*v);
# Dy = -neff*(Hx(i1) + Hx(i2) + Hx(i3) + Hx(i4))/4 - ...
#      (Hz(i3) + Hz(i4) - Hz(i1) - Hz(i2))./(j*2*k*h);
# Dz = ((Hy(i3) + Hy(i4) - Hy(i1) - Hy(i2))./(2*h) - ...
#       (Hx(i1) + Hx(i4) - Hx(i2) - Hx(i3))./(2*v))/(j*k);
#
Dx = Dx.reshape(nx, ny, order='F')
Dy = Dy.reshape(nx, ny, order='F')
Dz = Dz.reshape(nx, ny, order='F')

Ex = (eyy*Dx - exy*Dy)/edet
Ey = (exx*Dy - eyx*Dx)/edet
Ez = Dz/ezz

np.savetxt('Ex_Field_400nm.txt', np.abs(Ex))
np.savetxt('Ey_Field_400nm.txt', np.abs(Ey))
np.savetxt('Ez_Field_400nm.txt', np.abs(Ez))
np.savetxt('Structure.txt', eps)

Hx = Hx.reshape((nx+1), (ny+1), order='F')
Hy = Hy.reshape((nx+1), (ny+1), order='F')
Hz = Hz.reshape((nx+1), (ny+1), order='F')

x = np.linspace(0, dx[0]*(nx), nx+1)
y = np.linspace(0, dy[0, 0]*(ny), ny+1)

#plt.figure(4)
#plt.imshow(np.real(Ex.T), interpolation='nearest')
#
#plt.figure(5)
#plt.imshow(np.real(Ey.T), interpolation='nearest')

XX, YY = np.meshgrid(x, y)
# plt.contourf(XX, YY, np.real(Hx.T), interpolation = 'nearest')
plt.figure(3)
plt.title('Ex')
plt.imshow(np.abs(Ex.T))
plt.colorbar(orientation='horizontal')

plt.figure(4)
plt.title('Ey')
plt.imshow(np.abs(Ey.T))
plt.colorbar(orientation='horizontal')

plt.figure(5)
plt.title('Ez')
plt.imshow(np.abs(Ez.T))
plt.colorbar(orientation='horizontal')

plt.show()

