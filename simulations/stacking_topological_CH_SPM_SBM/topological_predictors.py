from scipy import linalg
import numpy as np
import pandas as pd
import networkx as nx

def compute_topological_features(A, edges):

    # create graph
    N = A.shape[0]
    id1, id2 = np.where(np.triu(A, 1))
    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    G.add_edges_from([(i, j) for i, j in np.concatenate((id1.reshape((len(id1),1)),id2.reshape((len(id2), 1))),axis=1)])

    # average degree (AD)
    ave_deg_net = np.sum(A) / N
    # variance of degree distribution (VD)
    var_deg_net = np.sqrt(np.sum(np.square(np.sum(A, axis=0) - ave_deg_net)) / (N - 1))
    # average (local) clustering coefficient (ACC)
    ave_clust_net = nx.average_clustering(G)

    # local number of triangles for i and j (LNT_i, LNT_j)
    numtriang_nodes_obj = nx.triangles(G)
    numtriang_nodes = []
    for nn in range(N):
        numtriang_nodes.append(numtriang_nodes_obj[nn])

    numtriang1_edges = []
    numtriang2_edges = []
    for ee in range(len(edges)):
        numtriang1_edges.append(numtriang_nodes[edges[ee][0]])
        numtriang2_edges.append(numtriang_nodes[edges[ee][1]])

    # Page rank values for i and j (PR_i, PR_j)
    page_rank_nodes_obj = nx.pagerank(G)
    page_rank_nodes = []
    for nn in range(N):
        page_rank_nodes.append(page_rank_nodes_obj[nn])

    page_rank1_edges = []
    page_rank2_edges = []
    for ee in range(len(edges)):
        page_rank1_edges.append(page_rank_nodes[edges[ee][0]])
        page_rank2_edges.append(page_rank_nodes[edges[ee][1]])

    # j-th entry of the personalized page rank of node i (PPR)
    page_rank_pers_nodes = []
    hot_vec = {}
    for nn in range(N):
        hot_vec[nn] = 0
    for nn in range(N):
        hot_vec_copy = hot_vec.copy()
        hot_vec_copy[nn] = 1
        page_rank_pers_nodes.append(nx.pagerank(G, personalization=hot_vec_copy))

    page_rank_pers_edges = []
    for ee in range(len(edges)):
        page_rank_pers_edges.append(page_rank_pers_nodes[edges[ee][0]][edges[ee][1]])

    # local clustering coefficients for i and j (LCC_i, LCC_j)
    clust_nodes_obj = nx.clustering(G)
    clust_nodes = []
    for nn in range(N):
        clust_nodes.append(clust_nodes_obj[nn])

    clust1_edges = []
    clust2_edges = []
    for ee in range(len(edges)):
        clust1_edges.append(clust_nodes[edges[ee][0]])
        clust2_edges.append(clust_nodes[edges[ee][1]])

    # average neighbor degrees for i and j (AND_i, AND_j)
    ave_neigh_deg_nodes_obj = nx.average_neighbor_degree(G)
    ave_neigh_deg_nodes = []
    for nn in range(N):
        ave_neigh_deg_nodes.append(ave_neigh_deg_nodes_obj[nn])

    ave_neigh_deg1_edges = []
    ave_neigh_deg2_edges = []
    for ee in range(len(edges)):
        ave_neigh_deg1_edges.append(ave_neigh_deg_nodes[edges[ee][0]])
        ave_neigh_deg2_edges.append(ave_neigh_deg_nodes[edges[ee][1]])

    # degree centralities for i and j (DC_i, DC_j)
    deg_cent_nodes_obj = nx.degree_centrality(G)
    deg_cent_nodes = []
    for nn in range(N):
        deg_cent_nodes.append(deg_cent_nodes_obj[nn])

    deg_cent1_edges = []
    deg_cent2_edges = []
    for ee in range(len(edges)):
        deg_cent1_edges.append(deg_cent_nodes[edges[ee][0]])
        deg_cent2_edges.append(deg_cent_nodes[edges[ee][1]])

    # eigenvector centralities for i and j (EC_i, EC_j)
    tr = 1
    toler = 1e-6
    eig_cent_nodes_obj = []
    while tr == 1:
        try:
            eig_cent_nodes_obj = nx.eigenvector_centrality(G, tol=toler)
            tr = 0
        except:
            toler = toler * 1e1

    eig_cent_nodes = []
    for nn in range(N):
        eig_cent_nodes.append(eig_cent_nodes_obj[nn])

    eig_cent1_edges = []
    eig_cent2_edges = []
    for ee in range(len(edges)):
        eig_cent1_edges.append(eig_cent_nodes[edges[ee][0]])
        eig_cent2_edges.append(eig_cent_nodes[edges[ee][1]])

    # Katz centralities for i and j (KC_i, KC_j)
    ktz_cent_nodes_obj = nx.katz_centrality_numpy(G)
    ktz_cent_nodes = []
    for nn in range(N):
        ktz_cent_nodes.append(ktz_cent_nodes_obj[nn])

    ktz_cent1_edges = []
    ktz_cent2_edges = []
    for ee in range(len(edges)):
        ktz_cent1_edges.append(ktz_cent_nodes[edges[ee][0]])
        ktz_cent2_edges.append(ktz_cent_nodes[edges[ee][1]])

    # Jaccard’s coefficient of neighbor sets of i, j (JC)
    jacc_coeff_obj = nx.jaccard_coefficient(G, edges)
    jacc_coeff_edges = []
    for uu, vv, jj in jacc_coeff_obj:
        jacc_coeff_edges.append([uu, vv, jj])
    df_jacc_coeff = pd.DataFrame(jacc_coeff_edges, columns=['i', 'j', 'jacc_coeff'])
    df_jacc_coeff['ind'] = df_jacc_coeff.index

    # resource allocation index of i, j (RA)
    res_alloc_ind_obj = nx.resource_allocation_index(G, edges)
    res_alloc_ind_edges = []
    for uu, vv, jj in res_alloc_ind_obj:
        res_alloc_ind_edges.append([uu, vv, jj])
    df_res_alloc_ind = pd.DataFrame(res_alloc_ind_edges, columns=['i', 'j', 'res_alloc_ind'])
    df_res_alloc_ind['ind'] = df_res_alloc_ind.index

    # Adamic/Adar index of i, j (AA)
    adam_adar_obj = nx.adamic_adar_index(G, edges)
    adam_adar_edges = []
    for uu, vv, jj in adam_adar_obj:
        adam_adar_edges.append([uu, vv, jj])
    df_adam_adar = pd.DataFrame(adam_adar_edges, columns=['i', 'j', 'adam_adar'])
    df_adam_adar['ind'] = df_adam_adar.index

    df_merge = pd.merge(df_jacc_coeff, df_res_alloc_ind, on=['ind', 'i', 'j'], sort=False)
    df_merge = pd.merge(df_merge, df_adam_adar, on=['ind', 'i', 'j'], sort=False)

    # preferential attachment (degree product) of i, j (PA)
    pref_attach_obj = nx.preferential_attachment(G, edges)
    pref_attach_edges = []
    for uu, vv, jj in pref_attach_obj:
        pref_attach_edges.append([uu, vv, jj])
    df_pref_attach = pd.DataFrame(pref_attach_edges, columns=['i', 'j', 'pref_attach'])
    df_pref_attach['ind'] = df_pref_attach.index

    # global features:
    # similarity of connections in the graph with respect to the node degree
    # degree assortativity (DA)
    deg_ass_net = nx.degree_assortativity_coefficient(G)
    # transitivity: fraction of all possible triangles present in G
    # network transitivity (clustering coefficient) (NT)
    transit_net = nx.transitivity(G)
    # network diameter (ND)
    try:
        diam_net = nx.diameter(G)
    except:
        diam_net = np.inf

    ave_deg_net = [ave_deg_net] * len(edges)
    var_deg_net = [var_deg_net] * len(edges)
    ave_clust_net = [ave_clust_net] * len(edges)
    deg_ass_net = [deg_ass_net] * len(edges)
    transit_net = [transit_net] * len(edges)
    diam_net = [diam_net] * len(edges)
    com_ne = []
    for ee in range(len(edges)):
        com_ne.append(len(sorted(nx.common_neighbors(G, edges[ee][0], edges[ee][1]))))

    # closeness centralities for i and j (CC_i, CC_j)
    closn_cent_nodes_obj = nx.closeness_centrality(G)
    closn_cent_nodes = []
    for nn in range(N):
        closn_cent_nodes.append(closn_cent_nodes_obj[nn])

    closn_cent1_edges = []
    closn_cent2_edges = []
    for ee in range(len(edges)):
        closn_cent1_edges.append(closn_cent_nodes[edges[ee][0]])
        closn_cent2_edges.append(closn_cent_nodes[edges[ee][1]])

    # shortest path between i, j (SP)
    short_Mat_aux = nx.shortest_path_length(G)
    short_Mat = {}
    for ss in range(N):
        value = next(short_Mat_aux)
        short_Mat[value[0]] = value[1]
    short_path_edges = []
    for ee in range(len(edges)):
        if edges[ee][1] in short_Mat[edges[ee][0]].keys():
            short_path_edges.append(short_Mat[edges[ee][0]][edges[ee][1]])
        else:
            short_path_edges.append(np.inf)

    # load centralities for i and j (LC_i, LC_j)
    load_cent_nodes_obj = nx.load_centrality(G, normalized=True)
    load_cent_nodes = []
    for nn in range(N):
        load_cent_nodes.append(load_cent_nodes_obj[nn])

    load_cent1_edges = []
    load_cent2_edges = []
    for ee in range(len(edges)):
        load_cent1_edges.append(load_cent_nodes[edges[ee][0]])
        load_cent2_edges.append(load_cent_nodes[edges[ee][1]])

    # shortest-path betweenness centralities for i and j (SPBC_i, SPBC_j)
    betw_cent_nodes_obj = nx.betweenness_centrality(G, normalized=True)
    betw_cent_nodes = []
    for nn in range(N):
        betw_cent_nodes.append(betw_cent_nodes_obj[nn])

    betw_cent1_edges = []
    betw_cent2_edges = []
    for ee in range(len(edges)):
        betw_cent1_edges.append(betw_cent_nodes[edges[ee][0]])
        betw_cent2_edges.append(betw_cent_nodes[edges[ee][1]])

    neigh_ = {}
    for nn in range(N):
        neigh_[nn] = np.where(A[nn, :])[0]

    df_pref_attach = []
    for ee in range(len(edges)):
        df_pref_attach.append(len(neigh_[edges[ee][0]]) * len(neigh_[edges[ee][1]]))

    try:
        U, sig, V = np.linalg.svd(A, full_matrices=False)
        S = np.diag(sig)
        Atilda = np.dot(U, np.dot(S, V))
        Atilda = np.array(Atilda)

        f_mean = lambda x: np.mean(x) if len(x) > 0 else 0
        # entry i, j in low rank approximation (LRA) via singular value decomposition (SVD)
        svd_edges = []
        # dot product of columns i and j in LRA via SVD for each pair of nodes i, j
        svd_edges_dot = []
        # average of entries i and j’s neighbors in low rank approximation
        svd_edges_mean = []
        for ee in range(len(edges)):
            svd_edges.append(Atilda[edges[ee][0], edges[ee][1]])
            svd_edges_dot.append(np.inner(Atilda[edges[ee][0], :], Atilda[:, edges[ee][1]]))
            svd_edges_mean.append(f_mean(Atilda[edges[ee][0], neigh_[edges[ee][1]]]))
    except:
        svd_edges = [0] * len(edges)
        svd_edges_dot = [0] * len(edges)
        svd_edges_mean = [0] * len(edges)

    # Leicht-Holme-Newman index of neighbor sets of i, j (LHN)
    f_LHN = lambda num, den: 0 if (num == 0 and den == 0) else float(num) / den
    LHN_edges = [f_LHN(num, den) for num, den in zip(np.array(com_ne), np.array(df_pref_attach))]

    try:
        U, sig, V = np.linalg.svd(A)
        S = linalg.diagsvd(sig, N, A.shape[1])
        S_trunc = S.copy()
        S_trunc[S_trunc < sig[int(np.ceil(np.sqrt(N)))]] = 0
        Atilda = np.dot(np.dot(U, S_trunc), V)
        Atilda = np.array(Atilda)

        f_mean = lambda x: np.mean(x) if len(x) > 0 else 0
        # an approximation of LRA (LRA-approx)
        svd_edges_approx = []
        # an approximation of dLRA (dLRA-approx)
        svd_edges_dot_approx = []
        # an approximation of mLRA (mLRA-approx)
        svd_edges_mean_approx = []
        for ee in range(len(edges)):
            svd_edges_approx.append(Atilda[edges[ee][0], edges[ee][1]])
            svd_edges_dot_approx.append(np.inner(Atilda[edges[ee][0], :], Atilda[:, edges[ee][1]]))
            svd_edges_mean_approx.append(f_mean(Atilda[edges[ee][0], neigh_[edges[ee][1]]]))
    except:
        svd_edges_approx = [0] * len(edges)
        svd_edges_dot_approx = [0] * len(edges)
        svd_edges_mean_approx = [0] * len(edges)

    # number of nodes (N)
    num_nodes = N
    # number of observed edges (OE)
    num_edges = int(np.sum(A) / 2)

    # construct a dictionary of the features
    d = {'i': edges[:, 0], 'j': edges[:, 1], 'com_ne': com_ne, 'ave_deg_net': ave_deg_net,
         'var_deg_net': var_deg_net, 'ave_clust_net': ave_clust_net, 'num_triangles_1': numtriang1_edges,
         'num_triangles_2': numtriang2_edges,
         'page_rank_pers_edges': page_rank_pers_edges, 'pag_rank1': page_rank1_edges, 'pag_rank2': page_rank2_edges,
         'clust_coeff1': clust1_edges, 'clust_coeff2': clust2_edges, 'ave_neigh_deg1': ave_neigh_deg1_edges,
         'ave_neigh_deg2': ave_neigh_deg2_edges,
         'eig_cent1': eig_cent1_edges, 'eig_cent2': eig_cent2_edges, 'deg_cent1': deg_cent1_edges,
         'deg_cent2': deg_cent2_edges, 'clos_cent1': closn_cent1_edges, 'clos_cent2': closn_cent2_edges,
         'betw_cent1': betw_cent1_edges, 'betw_cent2': betw_cent2_edges,
         'load_cent1': load_cent1_edges, 'load_cent2': load_cent2_edges, 'ktz_cent1': ktz_cent1_edges,
         'ktz_cent2': ktz_cent2_edges, 'pref_attach': df_pref_attach, 'LHN': LHN_edges, 'svd_edges': svd_edges,
         'svd_edges_dot': svd_edges_dot, 'svd_edges_mean': svd_edges_mean,
         'svd_edges_approx': svd_edges_approx, 'svd_edges_dot_approx': svd_edges_dot_approx,
         'svd_edges_mean_approx': svd_edges_mean_approx, 'short_path': short_path_edges, 'deg_assort': deg_ass_net,
         'transit_net': transit_net, 'diam_net': diam_net,
         'num_nodes': num_nodes, 'num_edges': num_edges}

    # construct a dataframe of the features
    df_feat = pd.DataFrame(data=d)
    df_feat['ind'] = df_feat.index
    df_feat = pd.merge(df_feat, df_merge, on=['ind', 'i', 'j'], sort=False)
    df_feat.drop(columns='ind', inplace=True)

    # cleaning features
    df_feat.loc[df_feat['short_path'] == np.inf, 'short_path'] = 1000 * max(df_feat.loc[~(df_feat['short_path'] == np.inf), 'short_path'])
    df_feat.loc[df_feat['diam_net'] == np.inf, 'diam_net'] = 1e6

    return df_feat
