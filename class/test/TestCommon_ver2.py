from scipy.linalg import eigvalsh 
import numpy as np
import DiagE

norb = 2
ns = 2
nk = 5
hlatt = np.zeros([norb,norb,ns,nk],dtype=complex,order='F')

for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                if iorb==jorb:
                    hlatt[iorb,jorb,js,ik] = 1.0+(ik+1)+(js+1)*0.1 
                else:
                    hlatt[iorb,jorb,js,ik] = 0.1 +(ik+1)+(js+1)*0.1 + ((iorb+1)+(jorb+1))*0.1 
hlatt2 = hlatt
E1 = np.zeros([norb],dtype=float,order='F')
E2 = np.zeros([norb],dtype=float,order='F')
print("hermitianeigen_dcmplx")
for ik in range(nk):
    for js in range(ns):
        tempmat1 = hlatt[:,:,js,ik]
        tempmat2 = hlatt[:,:,js,ik]
#       print(tempmat1)
#       print(tempmat2)  
#       tempmat1 = np.array(tempmat1.T,order='F')
        E1 = DiagE.common.hermitianeigen_dcmplx(tempmat1)
        print(tempmat1)
#       print(tempmat2)
        E2, tempmat2 = np.linalg.eig(tempmat2)
        print(tempmat2)
        E2 = np.linalg.eigvalsh(tempmat2)
#       print('/n')
        for i in range(norb):
            err = E1[i]-E2[i]
            if abs(err) >= 1.0e-8:
               print(err, E1[i],E2[i])
 
#H = np.array([[0,1],[1,0]],dtype=complex,order='F')
#E1 = DiagE.common.hermitianeigen_dcmplx(H)
#E2 = np.linalg.eigvalsh(H)
#
#print(E1)
#print(E2)
