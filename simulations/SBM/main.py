import scipy.io as spio
import numpy as np
import time 
import os.path
from SBM_DC_nested import compute_likelihood

# settings
networks = [line.rstrip('\n') for line in open('networks.txt')]
nested = [False]
DC = [True]
iters = 10

# paths
inputpath = '../../data/sparsified_matrices/'
suffix_list = ['_linkrem10', '_linkrem10_linkrem20noconn']
inputvar = 'matrices'
outputpath = './output/'
temppath = './TEMP/'
timepath = './time/'

for suffix in suffix_list:
    for nes in nested:
        for deg in DC:
        
            if deg==False:
                DC_str = ''
            else:
                DC_str = '_DC'
            
            if nes==False:
                nested_str = ''
            else:
                nested_str = '_N'
            
            for net in networks:
                x = (spio.loadmat(inputpath + net + suffix + '.mat'))[inputvar]
                N = x[1][0].shape[0]
                if N <= 100:
                    sweeps = 100
                elif N <= 1000:
                    sweeps = 50
                else:
                    sweeps = 10
    
                for i in range(iters):
                    outfile = outputpath + net + '_' + str(i+1) + DC_str + nested_str + suffix + '_scores.txt'
                    tempfile = temppath + net + '_' + str(i+1) + DC_str + nested_str + suffix + '.TEMP'
                    timefile = timepath + net + '_' + str(i+1) + DC_str + nested_str + suffix + '_time.txt'
                    if (not (os.path.isfile(outfile))) and (not (os.path.isfile(tempfile))):
                        os.mknod(tempfile)
                        
                        print(net + ' ' + str(i+1) + ' (DC = ' + str(deg) + ', Nested = ' + str(nes) + ') sweeps=' + str(sweeps) + ' ' + suffix, flush=True)
                        start = time.time()
                        probs = compute_likelihood(x[i][0], deg, nes, sweeps)
                        np.savetxt(outfile, probs, fmt=['%d','%d','%.18e'])
                        end = time.time()
                        print(end-start, flush=True)
                        
                        file = open(timefile, 'w')
                        file.write(str(end-start))
                        file.close()
                        
                        os.remove(tempfile)