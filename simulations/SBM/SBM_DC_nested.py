# Compute likelihood ratio of missing links through degree corrected or not and nested or not SBM.
#
# Adapted from Graph_tool project (https://graph-tool.skewed.de/)
# https://graph-tool.skewed.de/static/doc/demos/inference/inference.html#edge-prediction-as-binary-classification
# 10.03.2018 Umberto Michieli umberto.michieli94@gmail.com
# 25.05.2022 Alessandro Muscoloni

import numpy as np
import graph_tool.all as gt
    
# Input:
#   x - adjacency matrix     
#   DC - boolean (True use Degree-Corrected SBM, False not)
#   nested - boolean (True use nested SBM, False not)
#   sweeps - number of samplings from the partition space
#
# Output:
#   probs - numpy array of shape (M,3) where M is the number of missing links
#           with three columns indicating (node1, node2, likelihood ratio)

def compute_likelihood(x, DC, nested, sweeps):
    
    # create Graph
    g = gt.Graph(directed=False)
    g.add_edge_list(np.transpose(x.nonzero()))

    if nested==True:
        L = 10  # maximum hierarchy depth to L = 10
        state = gt.minimize_nested_blockmodel_dl(g, deg_corr=DC) # Initialize the Markov chain from the "ground state".
        # Before doing model averaging, we need to create a NestedBlockState by passing sampling = True.
        # We also want to increase the maximum hierarchy depth to L = 10.
        # We can do both of the above by copying.
        bs = state.get_bs()                     # Get hierarchical partition.
        bs += [np.zeros(1)] * (L - len(bs))     # Augment it to L = 10 with single-group levels.
        state = state.copy(bs=bs, sampling=True)
    else:
        state = gt.minimize_blockmodel_dl(g, deg_corr=DC)
    
    # find missing links
    missing = np.transpose(np.where(np.triu(x.todense()==0,1)))
    M = missing.shape[0]
    
    # define callback function
    probs = []
    def collect_edge_probs(s):
        p = np.empty(shape=(M,))
        for i in range(M):
            p[i] = s.get_edges_prob([missing[i]], entropy_args=dict(partition_dl=False))
        probs.append(p)

    # collect the probabilities for exactly 'sweeps' times
    gt.mcmc_equilibrate(state, force_niter=sweeps, mcmc_args=dict(niter=10), callback=collect_edge_probs)

    # average the probabilities and compute likelihood ratio
    probs = np.vstack(probs)
    pmax = probs.max(axis=0)
    probs = pmax + np.log(np.exp(probs-pmax).mean(axis=0))
    pmax = probs.max()
    psum = pmax + np.log(np.exp(probs-pmax).mean()) + np.log(len(probs))
    probs = np.exp(probs-psum)
    probs = np.hstack((missing, probs.reshape(M,1)))
    
    return probs
