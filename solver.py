from FRLM import benders
from FRLM import network
from FRLM import problem


from matplotlib import pyplot as plt
from itertools import product, combinations
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import numpy as np
import multiprocessing
import compress_json
import time
import os
import cplex

def get_expanded_network(idx, net, i, j, L, R, G_shortest_matrix):
        return idx, net.make_expanded_network(net.G, i, j, L, R, G_shortest_matrix)

def run(params):
    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()
            
    net = network.Net(dataDir=f"instance/{params['network_name']}", dist_multiplier=params['dist_multiplier'], allow_backward=params['allow_backward'], middle_node_sep_len=params['middle_node_sep_len'])
    mean_arc_dist = np.mean([v['length'] for i,adj in net.G.adj.items() for j,v in adj.items()])
    std_arc_dist = np.std([v['length'] for i,adj in net.G.adj.items() for j,v in adj.items()])
    num_nodes = len(net.G.nodes)
    num_arcs = len(net.G.edges)

    # find all possible OD pairs
    node_weight_l = [[i,net.G.nodes[i]['weight']] for i in net.G.nodes if net.G.nodes[i]['weight'] > 0]
    weighted_nodes = [i for i,w in sorted(node_weight_l,key=lambda l:l[1], reverse=True)][:params['num_node']]
    OD_pairs = [v for v in product(weighted_nodes, repeat=2) if v[0] != v[1]]
    if params['remove_near_OD']:
        OD_pairs = [(o,d) for o,d in OD_pairs if params['R'] < nx.shortest_path_length(net.G, o, d, weight='length')]
    possible_K = {k:v for k,v in enumerate(OD_pairs)}    
    possible_D_k = net.make_demand(possible_K)

    # find valid OD pair with expanded network
    nodes = list(net.G.nodes())
    G_shortest_matrix = {i:{j:nx.dijkstra_path_length(net.G, i, j, weight='length') for j in nodes} for i in nodes}

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-3) as pool:
        EN = dict(pool.starmap(get_expanded_network, [(key, net, value[0], value[1], params['L'], params['R']*params['dist_multiplier'], G_shortest_matrix) for key, value in possible_K.items()]))
    
    K = {}
    D_k = {}
    EN_k = {}
    for k,v in possible_K.items():
        if nx.has_path(EN[k]['net'], -1, -2):
            K[len(K)] = v
            D_k[len(D_k)] = possible_D_k[k]
            EN_k[len(K)] = EN[k]

    print(f"network : {params['network_name']}, R : {params['R']}, num_nodes : {params['num_node']}, |K| : {len(EN_k)}, expected |K| : {params['num_node']*(params['num_node']-1)})")


    start_time = time.time()

    make_prob_start_time = time.time()
    prob = problem.Problem(net, K, D_k, G_shortest_matrix, R=params['R']*params['dist_multiplier'], L=params['L'], B=params['B'], QT=params['QT']*params['dist_multiplier'],
                            W=params['W'], fst_q=params['fst_q'], alpha=params['alpha'], beta=params['beta'], theta=params['theta'],
                            delta=params['delta'], tau=params['tau'], decay_f=params['decay_f'], use_CT=params['use_CT'], use_BnC=params['use_BnC'],
                            num_preload_paths=params['num_preload_paths'], max_num_labels=params['max_num_labels'], use_length_limit=params['use_length_limit'])

    if params['use_multicut']:
        from FRLM import benders_multicut as benders
    else:
        from FRLM import benders
        
    model = benders.BD(prob, num_threads=params['num_threads'])
    make_prob_time = time.time() - make_prob_start_time

    solve_start_time = time.time()
    model.solve_master(time_limit=params['timelimit'])
    solve_time = time.time() - solve_start_time

    total_time = time.time() - start_time

    if model.master.solution.get_status_string() == 'aborted, no integer solution':
        obj_val = None
        solution = None
        gap = None
        covered_EN_paths_k, covered_full_paths_k = None, None
        detour_rate, time_increase_rate, decay_rate = None
        covered_demand_ratio, covered_flow_ratio = None, None
        covered_demand_origin_flow, covered_demand_detour_ratio = None, None
    else:
        obj_val = model.master.solution.get_objective_value()
        solution = model.get_solutions()
        # gap = cplex._internal._procedural.getmiprelgap(model.master._env._e, model.master._lp)
        gap = model.gap
        covered_EN_paths_k, covered_full_paths_k = benders.get_covered_paths(model)

        detour_rate = {}
        time_increase_rate = {}
        decay_rate = {}
        for k,v in covered_full_paths_k.items():
            sp_len = nx.dijkstra_path_length(prob.EN[k]['net'], prob.K[k][0], prob.K[k][1], weight='length')
            detour_rate[k] = v[1]/sp_len
            time_increase_rate[k] = v[2]/model.prob.shortest_time[k]    
            decay_rate[k] = v[3]/model.prob.DECAY[k][0]
        covered_demand_ratio = len(covered_EN_paths_k)/len(prob.K)
        covered_flow_ratio = obj_val/sum(model.prob.D_k.values())
        
        covered_demand_origin_flow = sum([D_k[int(k)] for k in covered_EN_paths_k.keys()])
        dfr = (covered_demand_origin_flow - obj_val)/covered_demand_origin_flow
        cost = {k:list(v[5]) for k,v in enumerate(prob.EN_INFO)}


    result = {'obj_val':obj_val, 'total_time':total_time, 'make_prob_time':make_prob_time, 'solve_time':solve_time,
            'gap':gap, 'bound':model.best_sub_obj_val, 'solution':solution, 'covered_EN_paths_k':covered_EN_paths_k, 'covered_full_paths_k':covered_full_paths_k,
            'shortest_time':model.prob.shortest_time, 'shortest_time_path':model.prob.shortest_time_path, 'shortest_time_full_path':model.prob.shortest_time_full_path,
            'detour_rate':detour_rate, 'time_increase_rate':time_increase_rate, 'decay_rate':decay_rate, 'D_k':model.prob.D_k,
            'lp_time':model.cb.lp_time if params['use_BnC'] else model.lp_time, 
            'cut_gen_time':model.cb.cut_gen_time if params['use_BnC'] else model.cut_gen_time,
            'labeling_time':model.cb.labeling_time if params['use_BnC'] else model.labeling_time,
            'cdr':covered_demand_ratio, 'cfr':covered_flow_ratio, 'dfr':dfr, 'demand':model.prob.D_k,
            'num_K':len(prob.K), 
            'num_path':{k:len(model.cb.paths[k]) if params['use_BnC'] else len(model.paths[k]) for k in prob.K},
            'num_bnb_node':cplex._internal._procedural.getnodecnt(model.master._env._e, model.master._lp),
            'num_called':model.cb.num_called if params['use_BnC'] else model.num_called,
            'num_cut':model.cb.num_cuts if params['use_BnC'] else model.num_cuts, 'total_demand':sum(model.prob.D_k.values()),
            'params':params, 'num_nodes':num_nodes, 'num_arcs':num_arcs,'mean_arc_dist':mean_arc_dist, 'std_arc_dist':std_arc_dist, 
            'cost':cost, 'CT':list(prob.CT)}

    save_dir = os.path.join('results',params['save_dir'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    compress_json.dump(result, os.path.join(save_dir,f"{params['save_fn']}.json.gz"),
                       json_kwargs={'default':np_encoder})

    # if not os.path.exists(os.path.join(save_dir, 'cut')):
    #     os.mkdir(os.path.join(save_dir, 'cut'))
    # with open(os.path.join(save_dir,'cut',f"{params['save_fn']}_cuts.pickle"), 'wb') as f:
    #     pickle.dump(model.cb.generated_cuts if params['use_BnC'] else model.generated_cuts, f, pickle.HIGHEST_PROTOCOL)

    # if not os.path.exists(os.path.join(save_dir, 'path')):
    #     os.mkdir(os.path.join(save_dir, 'path'))
    # with open(os.path.join(save_dir,'path',f"{params['save_fn']}_paths.pickle"), 'wb') as f:
    #     if params['use_BnC']:
    #         pickle.dump(model.cb.paths if params['use_BnC'] else model.paths, f, pickle.HIGHEST_PROTOCOL)