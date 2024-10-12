for i in range(1,100):
    for j in range(1,100):
        for k in range (1,100):
            for l in range (1,100):
                if i+j+k+l == i*j*k*l:
                    print(i,j,k,l)
