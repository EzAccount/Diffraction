jump = open('jump_density.dat', "w")
k = 0
one = 0
with open('NS_density.dat') as f:
    for line in f:
        k = k + 1
        if k > 200: one = 1
        temp_x, temp_y = [float(x) for x in line.split()]
        jump.write(str(temp_x)+' '+str(one)+'\n')
jump.close()
