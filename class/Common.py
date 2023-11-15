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