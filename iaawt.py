import numpy as np
import math
import skued as sk
import random

def iaawt(x, xdist=[], N=1, accerror=.001, error_change=100):
    
    if len(xdist)==0:
        xdist=x
        print("Taking the dsitribution of x for xdist")
    
    n = len(x)
    sgates = []
    
    numlevels = math.floor(math.log(n)/math.log(2))
    count = 0
    
    # wavelet decomp
    Y = sk.dtcwt(x, 'sym2', 'qshift2', level=numlevels) 
    Yl = Y[0]
    Yh = Y[1:len(Y)]
    
    #  real and amplitudes
    ampYh = []
    phaseYh = []
    for yh in Yh:
        ampYh.append(abs(yh))
        phaseYh.append(np.angle(yh))
        
    sortval = np.sort(x)
    stdval = np.std(sortval)
    
    sortval2 = np.sort(xdist)
    stdval2 = np.std(sortval2)
    
    for k in range(0,N):
        print("surrogate nr. ", k+1)
        z = sortval
        random.shuffle(z)
        z[0] = 0
        Z = sk.dtcwt(z, 'sym2', 'qshift2', level=numlevels) 
        Zh = Z[1:len(Z)]
        newphase = []
        for zh in Zh:
            newphase.append(np.angle(zh))
        
        amperror = [100]
        waverror = [100]
        counter = 0
        
        newZh = []
    
        while (amperror[counter] > accerror) & (waverror[counter] > accerror):
                
                # wavelet construction
                oldz = z
                newZh = [Yl]
                for i in range(0,len(ampYh)):
                  newZh.append(ampYh[i] * np.exp(newphase[i] * 1j))
                
                z = sk.idtcwt(newZh,'sym2', 'qshift2')
                wavdiff = np.mean(np.mean(abs(z.real - oldz.real)))
                waverror.append(wavdiff/stdval)
                
                # impose original values
                oldz = z
                data2sort = z[count:n]
                shuffind = np.argsort(data2sort)
                z[shuffind]  =  sortval2
                z[0:count] = 0
                ampdiff =np. mean(np.mean(abs(z.real-oldz.real)))
                amperror.append(ampdiff/stdval2)
                
                # Wavelet step
                nZ = sk.dtcwt(z, 'sym2', 'qshift2', level=numlevels) 
                nZh = nZ[1:len(nZ)]
                
                # get phases and imag
                newphase = []
                for nzh in nZh:
                    newphase.append(np.angle(nzh))
                
                toterror = amperror[counter+1] + waverror[counter+1]
                oldtoterr = amperror[counter] + waverror[counter]
                if abs((oldtoterr-toterror)/toterror) < (accerror/error_change):
                    amperror[counter+1] = -1
                    waverror[counter+1] = -1
                counter = counter + 1
                del nZh
          
        sgates.append(z)
        
    return sgates




