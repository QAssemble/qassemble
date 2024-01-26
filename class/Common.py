import numpy as np
def indexing(ntot, ndivision, divisionarray, flag, n1, n2):
    tmpsize = 1
    for size in divisionarray:
        tmpsize *= size

    if tmpsize != ntot:
        print('array_division wrong')
        return

    if flag == 1:
        n1 = n2[0]
        for ii in range(1, ndivision):
            tempcnt = 1
            for jj in range(ii):
                tempcnt *= divisionarray[jj]
            n1 += (n2[ii] ) * tempcnt
    else:
        n2_array = [0] * ndivision
        tempcnt = n1
        for ii in range(ndivision - 1):
            n2_array[ii] = tempcnt - ((tempcnt) // divisionarray[ii]) * divisionarray[ii]
            tempcnt = (tempcnt - n2_array[ii])//divisionarray[ii]
        n2_array[ndivision - 1] = tempcnt

        # Copy the values from the temporary array to the n2 output array
        for i in range(ndivision):
            n2[i] = n2_array[i]

    return n1, n2

def find_positions(array, value):
    positions = []
    for row_index, row in enumerate(array):
        for col_index, col_value in enumerate(row):
            if col_value == value:
                positions.append([row_index, col_index])
    return positions

def beta(T):
    return 1/(T*8.6173303*10**-5)

def save_file_locstc(iter : int,key:int,flag : int,fn : str,obj : np.ndarray):
    '''
    flag : 1 -> space file save
    flag : 0 -> problem file save
    '''
    if flag==1:
        fn = fn+'.'+str(iter)+'.'+str(key+1)+'.space.dat'
        with open(fn,"w") as file:
            for js in range(obj.shape[2]):
                for iorb in range(obj.shape[0]):
                    for jorb in range(obj.shape[0]):
                        file.write(f"{iorb} {jorb} {js} {np.real(obj[iorb,jorb,js])} {np.imag(obj[iorb,jorb,js])}\n")
    elif flag == 0:
        fn = fn+'.'+str(iter)+'.'+str(key+1)+'.problem.dat'
        with open(fn,"w") as file:
            for js in range(obj.shape[2]):
                for iorb in range(obj.shape[0]):
                    for jorb in range(obj.shape[0]):
                        file.write(f"{iorb} {jorb} {js} {np.real(obj[iorb,jorb,js])} {np.imag(obj[iorb,jorb,js])}\n")
    
    return None

def save_file_locdyn(iter : int,key:int,flag : int,fn : str,obj : np.ndarray):
    '''
    flag : 1 -> space file save
    flag : 0 -> problem file save
    '''
    if flag==1:
        fn = fn+'.'+str(iter)+'.'+str(key+1)+'.space.dat'
        with open(fn,"w") as file:
            for ifreq in range(obj.shape[3]):
                for js in range(obj.shape[2]):
                    for iorb in range(obj.shape[0]):
                        for jorb in range(obj.shape[0]):
                            file.write(f"{iorb} {jorb} {js} {ifreq} {np.real(obj[iorb,jorb,js,ifreq])} {np.imag(obj[iorb,jorb,js,ifreq])}\n")
    elif flag == 0:
        fn = fn+'.'+str(iter)+'.'+str(key+1)+'.problem.dat'
        with open(fn,"w") as file:
            for ifreq in range(obj.shape[3]):
                for js in range(obj.shape[2]):
                    for iorb in range(obj.shape[0]):
                        for jorb in range(obj.shape[0]):
                            file.write(f"{iorb} {jorb} {js} {ifreq} {np.real(obj[iorb,jorb,js,ifreq])} {np.imag(obj[iorb,jorb,js,ifreq])}\n")
    
    return None
