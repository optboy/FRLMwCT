from re import sub
from matplotlib import use
import networkx as nx
import cplex
import time
from networkx.algorithms.operators.unary import reverse
import numpy as np
import numba
import math
from concurrent.futures import ThreadPoolExecutor
import heapq

class BD:
    def __init__(self, prob, num_threads=5):
        self.start_time = time.time()
        if num_threads > 1:
            self.thread_executor = ThreadPoolExecutor(max_workers=num_threads)
        else:
            self.thread_executor = None

        self.cut_num = 0
        self.prob = prob

        self.sub_models = {k:self.set_sub_sol(k) for k in self.prob.K}

        self.set_master()
        
    def set_master(self):
        cpx = cplex.Cplex()
        
        # variables
        cpx.variables.add(names=[f"x_{i}_{w}" for i in self.prob.N for w in range(len(self.prob.W))],
                          types=['B' for i in self.prob.N for w in range(len(self.prob.W))])
        cpx.variables.add(names=[f"z_{k}" for k in self.prob.K], ub=[10e8 for k in self.prob.K])

        # constraints
        cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind = [f"x_{i}_{w}" for i in self.prob.N for w in range(len(self.prob.W))],
                    val = [self.prob.C_i[i][w] for i in self.prob.N for w in range(len(self.prob.W))]
                )
            ],
            senses=['L'],
            rhs=[self.prob.B],
            names=['budget']
        )

        cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind = [f"x_{i}_{w}" for w in range(len(self.prob.W))],
                    val = [1 for w in range(len(self.prob.W))]
                )
            for i in self.prob.N],
            senses=['L' for i in self.prob.N],
            rhs=[1 for i in self.prob.N],
            names=[f'station_type_{i}' for i in self.prob.N]
        )

        # objective
        cpx.objective.set_sense(cpx.objective.sense.maximize)

        cpx.objective.set_linear([(f"z_{k}", 1) for k in self.prob.K])

        if self.prob.use_BnC:
            cb = cpx.register_callback(LazyCutGenCallback)
            cb.prob = self.prob
            cb.paths = {k:[] for k in self.prob.K}
            cb.shortest_time = self.prob.shortest_time
            cb.use_hrstc = self.prob.use_hrstc
            cb.var_x_index = list(range(len(self.prob.N)*len(self.prob.W)))
            cb.thread_executor = self.thread_executor
            cb.sub_models = self.sub_models

            cb.start_time = self.start_time
            cb.cut_gen_time = 0
            cb.lp_time = 0
            cb.labeling_time = 0
            cb.num_called = 0
            cb.num_cuts = 0
            cb.generated_cuts = []

            self.cb = cb

        else:
            cpx.set_log_stream(None)
            cpx.set_error_stream(None)
            cpx.set_warning_stream(None)
            cpx.set_results_stream(None)
            
            self.paths = {k:[] for k in self.prob.K}
            self.var_x_index = list(range(len(self.prob.N)*len(self.prob.W)))
            self.cut_gen_time = 0
            self.lp_time = 0
            self.labeling_time = 0
            self.num_called = 0
            self.num_cuts = 0
            self.generated_cuts = []

        self.master = cpx

    def set_sub_sol(self, k):
        cpx = cplex.Cplex()
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_results_stream(None)

        if self.prob.use_hrstc[k]:
            # variables
            cpx.variables.add(names=[f"alpha_{i}" for i in self.prob.N])
            cpx.variables.add(names=[f"beta"])
            # objective
            cpx.objective.set_sense(cpx.objective.sense.minimize)
            return cpx
        else:
            # variables
            cpx.variables.add(names=[f"alpha_{i}_{w}" for w in range(len(self.prob.W)) for i in self.prob.N])
            cpx.variables.add(names=[f"beta"])
            # objective
            cpx.objective.set_sense(cpx.objective.sense.minimize)
            return cpx

    def solve_master(self, time_limit=3600):
        if self.prob.use_BnC:
            self.cb.time_limit = time_limit
            self.master.parameters.timelimit.set(time_limit)

            self.master.solve()
        else:
            self.solve_master_iterative(time_limit)

    def solve_master_iterative(self, time_limit=3600):
        
        self.bounds = []

        self.time_limit = time_limit

        while True:
            if time.time() - self.start_time > self.time_limit:
                self.master.solve()

                bound = 0
                for idx,sub_model in self.sub_models.items():
                    if sub_model.solution.get_status() == 1:
                        bound += sub_model.solution.get_objective_value()
                    else:
                        sub_model.solve()
                        bound += sub_model.solution.get_objective_value()
                self.bounds.append(bound)

                self.best_sub_obj_val = max(self.bounds)
                self.gap = round(100*(master_obj_val - self.best_sub_obj_val)/master_obj_val, 2)
                break
            else:
                self.master.solve()

                self.num_called += 1

                g_star = get_corrected_charging_pattern(self.prob, self.master.solution.get_values())
                
                alpha_k, beta_k, cut_gen_time, labeling_time, lp_solve_time = cut_generation(self.prob, self.master.solution.get_values(), 
                            self.sub_models, self.paths, self.prob.shortest_time, self.prob.use_hrstc, self.start_time, self.time_limit, self.thread_executor, self.prob.num_preload_paths)

                self.lp_time += lp_solve_time
                self.cut_gen_time += cut_gen_time
                self.labeling_time += labeling_time

                corrected_alpha_k = recalculate_alpha(self.prob, alpha_k, beta_k, g_star, self.prob.use_hrstc)

                cut_k = {}
                for k in self.prob.K:
                    pi_l_k = get_pi_k(self.prob, corrected_alpha_k, k)

                    cut = get_cut_k(self.prob, pi_l_k, k)
                    self.generated_cuts.append((cut, beta_k[k], self.master.solution.get_values()))
                    cut_k[k] = cut

                bound = 0
                for idx,sub_model in self.sub_models.items():
                    if sub_model.solution.get_status() == 1:
                        bound += sub_model.solution.get_objective_value()
                    else:
                        sub_model.solve()
                        bound += sub_model.solution.get_objective_value()
                self.bounds.append(bound)

                master_obj_val = self.master.solution.get_objective_value()
                self.best_sub_obj_val = max(self.bounds)
                self.gap = round(100*(master_obj_val - self.best_sub_obj_val)/master_obj_val, 2)
                print(f'time : {round(time.time() - self.start_time)}\tmaster : {master_obj_val}\tbound : {self.best_sub_obj_val}\tgap : {self.gap}%')

                if master_obj_val - self.best_sub_obj_val <= 1e-3:
                    break
                
                for k, cut in cut_k.items():
                    self.master.linear_constraints.add(
                                    lin_expr=[cut],
                                    senses=['L'],
                                    rhs=[beta_k[k]],
                                    names=[f'user_cut_{self.num_cuts}']
                                )

                    self.num_cuts += 1

    def get_master_sol(self):
        return [self.master.solution.get_values(f"x_{i}_{w}") for i in self.prob.N for w in range(len(self.prob.W))]

    def get_solutions(self):
        installed_nodes = [i for i in self.prob.N for w in range(len(self.prob.W)) if self.master.solution.get_values(f"x_{i}_{w}") > 0.8]
        installed_types = [w for i in self.prob.N for w in range(len(self.prob.W)) if self.master.solution.get_values(f"x_{i}_{w}") > 0.8]
        return installed_nodes, installed_types

def get_flow(prob, k, nodes, station_types, get_time=False, shortest_time=None):
    time = 0
    b = prob.R/2
    D = nodes[-1]
    N = list(prob.EN[k]['net'].nodes)

    for i_idx, node in enumerate(nodes[:-1]):
        i, j = node, nodes[i_idx+1]
        q = prob.Q[station_types[i_idx]]

        dist_ij = prob.EN_INFO[k][4][N.index(i)][N.index(j)]

        if b >= dist_ij:
            if j == D:
                time = time + dist_ij + prob.QT + round(q*prob.CT[int(max(b, prob.R/2+dist_ij))],3) - round(q*prob.CT[int(b)],3)
                b = max(b, prob.R/2 + dist_ij) - dist_ij
            else:
                time = time + dist_ij
                b = b - dist_ij
        else:
            if j == D:
                time = time + dist_ij + prob.QT + round(q*prob.CT[int(prob.R/2 + dist_ij)],3) - round(q*prob.CT[int(b)],3) 
                b = prob.R/2
            else:
                time = time + dist_ij + prob.QT + round(q*prob.CT[int(dist_ij)],3) - round(q*prob.CT[int(b)],3)
                b = 0

    if get_time:
        return time
    else:
        if max(int(time-shortest_time[k]),0) < len(prob.DECAY[k]):
            return prob.DECAY[k][max(int(time-shortest_time[k]),0)]
        else:
            return 0

def get_PnG_from_sol(prob, path):
    P = []
    G = []
    for i in path:
        if i < 0:
            P.append(i)
            G.append(0)
        else:
            P.append(i % len(prob.N))
            G.append(i // len(prob.N))
    return P, G

def cut_generation_v1(prob, x, sub_models, paths, shortest_time, use_hrstc, start_time, time_limit, thread_executor=5):
    cut_gen_start_time = time.time()
    done = False

    g_star = get_corrected_charging_pattern(prob,x)

    lp_solve_time = 0
    lp_start_time = time.time()
    for k in prob.K:
        update_sub_model(prob, k, x, g_star, sub_models, paths, shortest_time, use_hrstc[k])
    lp_solve_time += time.time() - lp_start_time

    labeling_time = 0
    while not done:        
        lp_start_time = time.time()
        alpha_k = []
        beta_k = []
        for k in prob.K:
            alpha, beta = get_sub_sol(prob, k, sub_models, use_hrstc[k])
            alpha_k.append(alpha)
            beta_k.append(beta)
        lp_solve_time += time.time() - lp_start_time

        if time.time() - start_time > time_limit:
            return alpha_k, beta_k, time.time()-cut_gen_start_time, labeling_time, lp_solve_time

        violated = False

        EN_list = [[O, D, N, R, ADJ_MAT, np.array(ALPHA), CT, QT, CS, g_star, Q, UH] 
                    for (O, D, N, R, ADJ_MAT, DECAY, CT, QT, CS, Q), ALPHA, UH in zip(prob.EN_INFO, alpha_k, use_hrstc)]
        
        labeling_start_time = time.time()

        all_results = list(thread_executor.map(lambda p: labeling_numba(*p), EN_list))

        labeling_time += time.time() - labeling_start_time
        
        labels = [l for l,p in all_results]
        path = [p for l,p in all_results]
        flow = [[prob.DECAY[k][int(t-shortest_time[k])] for a,t in labels[k]] for k in prob.K]

        for k in prob.K:
            for p_idx,p in enumerate(path[k]):
                label = labels[k][p_idx]
                if use_hrstc[k]:
                    # if p not in paths[k] and calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                    if calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                        violated = True
                        add_path(prob, k, x, g_star, sub_models, paths, p, shortest_time, use_hrstc)
                        paths[k].append(p)
                else:
                    P,G = get_PnG_from_sol(prob, p)
                    # if (P,G) not in [(p_k,g) for p_k,g,f in paths[k]] and calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                    if calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                        violated = True
                        add_path(prob, k, x, g_star, sub_models, paths, (P,G,flow[k][p_idx]), shortest_time, use_hrstc)
                        paths[k].append((P,G,flow[k][p_idx]))
        if not violated:
            done = True

    cut_gen_time = time.time() - cut_gen_start_time
    return alpha_k, beta_k, cut_gen_time, labeling_time, lp_solve_time


def cut_generation_v2(prob, x, sub_models, paths, shortest_time, use_hrstc, start_time, time_limit, thread_executor=5):
    cut_gen_start_time = time.time()
    done = False

    g_star = get_corrected_charging_pattern(prob,x)

    lp_solve_time = 0
    lp_start_time = time.time()
    for k in prob.K:
        update_sub_model(prob, k, x, g_star, sub_models, paths, shortest_time, use_hrstc[k])
    lp_solve_time += time.time() - lp_start_time

    labeling_time = 0
    while not done:
        
        lp_start_time = time.time()
        alpha_k = []
        beta_k = []
        for k in prob.K:
            alpha, beta = get_sub_sol(prob, k, sub_models, use_hrstc[k])
            alpha_k.append(alpha)
            beta_k.append(beta)
        lp_solve_time += time.time() - lp_start_time

        if time.time() - start_time > time_limit:
            return alpha_k, beta_k, time.time()-cut_gen_start_time, labeling_time, lp_solve_time

        violated = False

        for sep_K in [prob.fwd_K, prob.bwd_K]:
            fwd_EN_INFO = [prob.EN_INFO[k] for k in sep_K]
            fwd_alpha_k = [alpha_k[k] for k in sep_K]
            fwd_use_hustc = [use_hrstc[k] for k in sep_K]
            EN_list = [[O, D, N, R, ADJ_MAT, np.array(ALPHA), CT, QT, CS, g_star, Q, UH] 
                        for (O, D, N, R, ADJ_MAT, DECAY, CT, QT, CS, Q), ALPHA, UH in zip(fwd_EN_INFO, fwd_alpha_k, fwd_use_hustc)]
            
            labeling_start_time = time.time()
            all_results = list(thread_executor.map(lambda p: labeling_numba(*p), EN_list))
            labeling_time += time.time() - labeling_start_time
            
            labels = [l for l,p in all_results]
            path = [p for l,p in all_results]
            flow = [[prob.DECAY[k][int(t-shortest_time[k])] for a,t in labels[idx]] for idx,k in enumerate(sep_K)]

            for idx,(k,v) in enumerate(sep_K.items()):
                for p_idx,p in enumerate(path[idx]):
                    rev_k = prob.K_idx[(v[1],v[0])]
                    label = labels[idx][p_idx]
                    if use_hrstc[k]:
                        # if p not in paths[k] and calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                        if calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                            violated = True
                            add_path(prob, k, x, g_star, sub_models, paths, p, shortest_time, use_hrstc)
                            paths[k].append(p)

                        rev_path = [-1] + [i for i in reversed(p)][1:-1] + [-2]
                        rev_ct = [g_star[i] for i in rev_path]
                        rev_dual = sum([alpha_k[rev_k][i] for i in rev_path[1:-1]])
                        rev_time = get_flow(prob, rev_k, rev_path, rev_ct, get_time=True)
                        if rev_path not in paths[rev_k] and calculate_reduced_cost(prob, rev_k, (rev_dual,rev_time), beta_k[rev_k], shortest_time) <= -1e-3:
                        # if calculate_reduced_cost(prob, rev_k, (rev_dual,rev_time), beta_k[rev_k], shortest_time) <= -1e-3:
                            violated = True
                            add_path(prob, rev_k, x, g_star, sub_models, paths, rev_path, shortest_time, use_hrstc)
                            paths[rev_k].append(rev_path)
                    else:
                        P,G = get_PnG_from_sol(prob, p)
                        # if (P,G) not in [(p_k,g) for p_k,g,f in paths[k]] and calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                        if calculate_reduced_cost(prob, k, label, beta_k[k], shortest_time) <= -1e-3:
                            violated = True
                            add_path(prob, k, x, g_star, sub_models, paths, (P,G,flow[idx][p_idx]), shortest_time, use_hrstc)
                            paths[k].append((P,G,flow[idx][p_idx]))
                        
                        rev_path = [-1] + [i for i in reversed(P)][1:-1] + [-2]
                        rev_ct = [i for i in reversed(G)]
                        rev_dual = sum([alpha_k[rev_k][i] for i in rev_path[1:-1]])
                        rev_time = get_flow(prob, rev_k, rev_path, rev_ct, get_time=True)
                        rev_flow = get_flow(prob, rev_k, rev_path, rev_ct, get_time=False, shortest_time=shortest_time)
                        if (rev_path,rev_ct) not in [(p_k,g) for p_k,g,f in paths[rev_k]] and calculate_reduced_cost(prob, rev_k, (rev_dual,rev_time), beta_k[rev_k], shortest_time) <= -1e-3:
                        # if calculate_reduced_cost(prob, rev_k, (rev_dual,rev_time), beta_k[rev_k], shortest_time) <= -1e-3:
                            violated = True
                            add_path(prob, rev_k, x, g_star, sub_models, paths, (rev_path, rev_ct, rev_flow), shortest_time, use_hrstc)
                            paths[rev_k].append((rev_path, rev_ct, rev_flow))

        if not violated:
            done = True

    cut_gen_time = time.time() - cut_gen_start_time
    return alpha_k, beta_k, cut_gen_time, labeling_time, lp_solve_time


# cut_generation v3
def cut_generation(prob, x, sub_models, paths, shortest_time, use_hrstc, start_time, time_limit, thread_executor=5, num_preload_paths=10):
    cut_gen_start_time = time.time()
    done = False

    g_star = get_corrected_charging_pattern(prob, x)

    lp_solve_time = 0
    lp_start_time = time.time()
    for k in prob.K:
        update_sub_model(prob, k, x, g_star, sub_models, paths, shortest_time, use_hrstc[k])
    lp_solve_time += time.time() - lp_start_time

    need_solve_K = [True] * len(prob.K)

    # The latest solutions
    alpha_k_dict = {}
    beta_k_dict = {}

    labeling_time = 0
    while not done:

        # Solve forward demands
        lp_start_time = time.time()
        fwd_alpha_k = []
        fwd_beta_k = []
        solved_K = []
        for k in prob.fwd_K:
            if need_solve_K[k]:
                alpha, beta = get_sub_sol(prob, k, sub_models, use_hrstc[k])
                fwd_alpha_k.append(alpha)
                fwd_beta_k.append(beta)

                alpha_k_dict[k] = alpha
                beta_k_dict[k] = beta

                solved_K.append(k)
                need_solve_K[k] = False

        lp_solve_time += time.time() - lp_start_time

        violated = False

        fwd_EN_INFO = [prob.EN_INFO[k] for k in solved_K]
        fwd_use_hustc = [use_hrstc[k] for k in solved_K]
        EN_list = [[O, D, N, R, ADJ_MAT, np.array(ALPHA), CT, QT, CS, g_star, Q, UH, False, num_preload_paths, np.array(DECAY), BETA, shortest_time[k], prob.max_num_labels]
                   for (O, D, N, R, ADJ_MAT, DECAY, CT, QT, CS, Q), ALPHA, BETA, UH, k in
                   zip(fwd_EN_INFO, fwd_alpha_k, fwd_beta_k, fwd_use_hustc, solved_K)]

        labeling_start_time = time.time()
        all_results = list(thread_executor.map(lambda p: labeling_numba_preload_path(*p), EN_list))
        labeling_time += time.time() - labeling_start_time

        labels = [l for l, p in all_results]
        path = [p for l, p in all_results]
        flow = [[prob.DECAY[k][int(t - shortest_time[k])] for a, t in labels[idx]] for idx, k in enumerate(solved_K)]

        for idx, k in enumerate(solved_K):
            v = prob.K[k]
            for p_idx, p in enumerate(path[idx]):
                
                P, G = get_PnG_from_sol(prob, p)
                if prob.use_length_limit:
                    if get_path_length(prob, P) >= prob.G_shortest_matrix[P[1]][P[-2]] * prob.L:
                        # print(get_path_length(prob, P), prob.G_shortest_matrix[P[1]][P[-2]] * prob.L)
                        continue

                rev_k = prob.K_idx[(v[1], v[0])]
                label = labels[idx][p_idx]
                if use_hrstc[k]:
                    if calculate_reduced_cost(prob, k, label, fwd_beta_k[idx], shortest_time) <= -1e-3:
                        violated = True
                        add_path(prob, k, x, g_star, sub_models, paths, p, shortest_time, use_hrstc)
                        paths[k].append(p)
                        need_solve_K[k] = True

                        rev_path = [-1] + [i for i in reversed(p)][1:-1] + [-2]
                        add_path(prob, rev_k, x, g_star, sub_models, paths, rev_path, shortest_time, use_hrstc)
                        paths[rev_k].append(rev_path)
                        need_solve_K[rev_k] = True

                else:
                    # P, G = get_PnG_from_sol(prob, p)
                    if calculate_reduced_cost(prob, k, label, fwd_beta_k[idx], shortest_time) <= -1e-3:
                        violated = True
                        add_path(prob, k, x, g_star, sub_models, paths, (P, G, flow[idx][p_idx]), shortest_time,
                                 use_hrstc)
                        paths[k].append((P, G, flow[idx][p_idx]))
                        need_solve_K[k] = True

                        rev_path = [-1] + [i for i in reversed(P)][1:-1] + [-2]
                        rev_ct = [i for i in reversed(G)]
                        rev_flow = get_flow(prob, rev_k, rev_path, rev_ct, get_time=False, shortest_time=shortest_time)
                        add_path(prob, rev_k, x, g_star, sub_models, paths, (rev_path, rev_ct, rev_flow),
                                 shortest_time, use_hrstc)
                        paths[rev_k].append((rev_path, rev_ct, rev_flow))
                        need_solve_K[rev_k] = True

        if violated:
            # Skip backward demands if any path was added
            continue

        # Solve backward demands
        lp_start_time = time.time()
        bwd_alpha_k = []
        bwd_beta_k = []
        solved_K = []
        for k in prob.bwd_K:
            if need_solve_K[k]:
                alpha, beta = get_sub_sol(prob, k, sub_models, use_hrstc[k])
                bwd_alpha_k.append(alpha)
                bwd_beta_k.append(beta)

                alpha_k_dict[k] = alpha
                beta_k_dict[k] = beta

                solved_K.append(k)
                need_solve_K[k] = False

        lp_solve_time += time.time() - lp_start_time

        violated = False

        bwd_EN_INFO = [prob.EN_INFO[k] for k in solved_K]
        bwd_use_hustc = [use_hrstc[k] for k in solved_K]
        EN_list = [[O, D, N, R, ADJ_MAT, np.array(ALPHA), CT, QT, CS, g_star, Q, UH, False, num_preload_paths, np.array(DECAY), BETA, shortest_time[k], prob.max_num_labels]
                   for (O, D, N, R, ADJ_MAT, DECAY, CT, QT, CS, Q), ALPHA, BETA, UH, k in
                   zip(bwd_EN_INFO, bwd_alpha_k, bwd_beta_k, bwd_use_hustc, solved_K)]

        labeling_start_time = time.time()
        all_results = list(thread_executor.map(lambda p: labeling_numba_preload_path(*p), EN_list))
        labeling_time += time.time() - labeling_start_time

        labels = [l for l, p in all_results]
        path = [p for l, p in all_results]
        flow = [[prob.DECAY[k][int(t - shortest_time[k])] for a, t in labels[idx]] for idx, k in enumerate(solved_K)]

        for idx, k in enumerate(solved_K):
            for p_idx, p in enumerate(path[idx]):

                P, G = get_PnG_from_sol(prob, p)
                if prob.use_length_limit:
                    if get_path_length(prob, P) >= prob.G_shortest_matrix[P[1]][P[-2]] * prob.L:
                        # print(get_path_length(prob, P), prob.G_shortest_matrix[P[1]][P[-2]] * prob.L)
                        continue
                    
                label = labels[idx][p_idx]
                if use_hrstc[k]:
                    if calculate_reduced_cost(prob, k, label, bwd_beta_k[idx], shortest_time) <= -1e-3:
                        violated = True
                        add_path(prob, k, x, g_star, sub_models, paths, p, shortest_time, use_hrstc)
                        paths[k].append(p)
                        need_solve_K[k] = True
                else:
                    # P, G = get_PnG_from_sol(prob, p)
                    if calculate_reduced_cost(prob, k, label, bwd_beta_k[idx], shortest_time) <= -1e-3:
                        violated = True
                        add_path(prob, k, x, g_star, sub_models, paths, (P, G, flow[idx][p_idx]), shortest_time,
                                 use_hrstc)
                        paths[k].append((P, G, flow[idx][p_idx]))
                        need_solve_K[k] = True

        if not violated:
            done = True

        if time.time() - start_time > time_limit:
            alpha_k = [alpha_k_dict[k] for k in prob.K]
            beta_k = [beta_k_dict[k] for k in prob.K]
            return alpha_k, beta_k, time.time() - cut_gen_start_time, labeling_time, lp_solve_time

    alpha_k = [alpha_k_dict[k] for k in prob.K]
    beta_k = [beta_k_dict[k] for k in prob.K]
    cut_gen_time = time.time() - cut_gen_start_time

    return alpha_k, beta_k, cut_gen_time, labeling_time, lp_solve_time


def get_sub_sol(prob, k, sub_models, use_hrstc):
    
    sub_models[k].solve()

    # if use_hrstc:
    #     return sub_models[k].solution.get_values([f"alpha_{i}" for i in prob.N]), sub_models[k].solution.get_values(f"beta")
    # else:
    #     return sub_models[k].solution.get_values([f"alpha_{i}_{w}" for w in range(len(prob.W)) for i in prob.N]), sub_models[k].solution.get_values(f"beta")

    sol = sub_models[k].solution.get_values()
    return sol[:-1], sol[-1]


def update_sub_model(prob, k, x, g_star, sub_models, paths, shortest_time, use_hrstc):
    if use_hrstc:
        for p_idx,p in enumerate(paths[k]):
            flow = float(get_flow(prob, k, p, [g_star[i] for i in p], get_time=False, shortest_time=shortest_time))
            sub_models[k].linear_constraints.set_rhs(f'const_{p_idx}', flow)

        sub_models[k].objective.set_linear([(f"alpha_{i}", x[i*len(prob.W)+g_star[i]]) for i in prob.N] + [('beta', 1)])
    else:
        sub_models[k].objective.set_linear([(f"alpha_{i}_{w}", x[i*len(prob.W)+w]) for i in prob.N for w in range(len(prob.W))] + [('beta', 1)])

def add_path(prob, k, x, g_star, sub_models, paths, new_path, shortest_time, use_hrstc):
    if use_hrstc[k]:
        sub_models[k].linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind = [f"alpha_{i}" for i in new_path[1:-1]] + ['beta'],
                    val = [1 for i in new_path[1:-1]] + [1]
                )
            ],
            senses=['G'],
            rhs=[float(get_flow(prob, k, new_path, [g_star[i] if i >= 0 else 0 for i in new_path], get_time=False, shortest_time=shortest_time))],
            names=[f'const_{len(paths[k])}']
        )
    else:
        p,g,f = new_path
        sub_models[k].linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind = [f"alpha_{i}_{w}" for i,w in zip(p[1:-1],g[1:-1])] + ['beta'],
                        val = [1 for i,w in zip(p[1:-1],g[1:-1])] + [1]
                    )
                ],
                senses=['G'],
                rhs=[float(f)],
                names=[f'const_{len(paths[k])}']
            )


@numba.njit(cache=True, parallel=False, nogil=True, fastmath=True)
def dijkstra(n, ADJ, cost, source, target, terminate_at_target=True):
    dist = {}
    prev = np.ones(n, dtype=np.int64)

    queue = [(0.0,source,-1)]

    heapq.heapify(queue)

    # dist[source] = 0.0

    while queue:
        d,i,p_i = heapq.heappop(queue)

        if i in dist:
            continue

        dist[i] = d
        prev[i] = p_i

        if terminate_at_target and i == target:
            break

        for j in ADJ[i]:
            if j >= 0 and j not in dist:
                c_ij = cost[i,j]
                heapq.heappush(queue, (d + c_ij, j, i))
                prev[j] = i

    path = [target]
    if target in dist: # the shortest path is found
        j = target
        while True:
            i = prev[j]
            path.insert(0, i)
            if i == source:
                break
            j = i
        return dist[target], path, dist
    else:
        return 0.0, path, dist
            



@numba.njit(cache=True, parallel=False, nogil=True, fastmath=True)
def ksp(n, ADJ, cost, source, target, K=5):

    ADJ_COPY = [
        l.copy() for l in ADJ
    ]

    dist, path, distances = dijkstra(n, ADJ, cost, source, target, terminate_at_target=False)

    A = [(dist, path)]
    B = [(0.0, [1,2,3])]

    heapq.heapify(B)
    heapq.heappop(B)

    # print(A)

    for kth in range(1,K):
        path = A[-1][1]
        for idx in range(0, len(path)-1):
            node_spur = path[idx] 
            path_root = path[:idx+1]
            dist_path_root = sum([cost[i,j] for (i,j) in zip(path_root[:-1], path_root[1:])])

            edges_removed = set()
            for cur_dist, cur_path in A:
                if len(cur_path) > idx and path_root == cur_path[:idx+1]:
                    adj_nodes = ADJ[cur_path[idx]]
                    ADJ[cur_path[idx]] = adj_nodes[adj_nodes != cur_path[idx+1]]
                    edges_removed.add(cur_path[idx])

            if len(path_root) > 1:
                for i in path_root[:-1]:
                    ADJ[i] = np.zeros(0, dtype=np.int64)
                    edges_removed.add(i)

            dist_path_spur, path_spur, _ = dijkstra(n, ADJ, cost, node_spur, target)

            if len(path_spur) > 1:
                path_total = path_root[:-1] + path_spur
                dist_total = dist_path_root + dist_path_spur
                # potential_k = (dist_total, path_total)
                for d,p in B:
                    if d == dist_total and p == path_total:
                        break
                else:
                    heapq.heappush(B, (dist_total, path_total))
                # if kth >= 0:
                #     print(f'  {path_total=} {dist_total=} {dist_path_root=} {dist_path_spur=} {node_spur=}')

            for i in edges_removed:
                ADJ[i] = ADJ_COPY[i]

        if len(B) > 0:
            best_dist, best_path = heapq.heappop(B)
            A.append((best_dist, best_path))

            # print(f'{kth}: {best_path}')
        else:
            break

    return A



@numba.njit(cache=True, parallel=False, nogil=True, fastmath=True)
def labeling_numba(O, D, N, R, ADJ_MAT, ALPHA, CT, QT, CS, W, CTR, use_hrstc, fst_q, preload_paths=None, DECAY=None, beta=0.0, sp_time=0.0, max_num_labels=-1):
    # Label definition:
    # [alpha, time, battery, distance, previous nodes]

    # label initialization 
    A = []      # float64, alpha
    T = []      # float64, time
    B = []      # float64, battery
    # DIST = []   # float64, distance
    PREV = []   # int64, previous visted nodes
    TREAT = []  # bool, indicates treated labels
    VALID = []  # bool, indicates removed labels

    N_IDX = {}
    for (idx,i) in enumerate(N):
        N_IDX[i] = idx

    ADJ_LIST = [
        [N[n_idx] for n_idx, val in enumerate(ADJ_MAT[idx]) if val >= 0]
        for idx, i in enumerate(N)
    ]

    for i in range(0, len(N)):
        A.append([0.0]), A[-1].pop()
        T.append([0.0]), T[-1].pop()
        B.append([0.0]), B[-1].pop()
        # DIST.append([0.0]), DIST[-1].pop()
        PREV.append([numba.int64(0)]), PREV[-1].pop()
        TREAT.append([numba.boolean(True)]), TREAT[-1].pop()
        VALID.append([numba.boolean(True)]), VALID[-1].pop()

    def add_label(n, a, t, b, prev):
        n_idx = N_IDX[n]
        for idx, valid in enumerate(VALID[n_idx]):
            if not valid:
                A[n_idx][idx] = a
                T[n_idx][idx] = t
                B[n_idx][idx] = b
                # DIST[n_idx][idx] = dist
                PREV[n_idx][idx] = prev
                TREAT[n_idx][idx] = False
                VALID[n_idx][idx] = True
                return 
        A[n_idx].append(a)
        T[n_idx].append(t)
        B[n_idx].append(b)
        # DIST[n_idx].append(dist)
        PREV[n_idx].append(prev)
        TREAT[n_idx].append(False)
        VALID[n_idx].append(True)

    def remove_label(n, k):
        n_idx = N_IDX[n]
        # n_idx = np.where(N == n)[0][0]
        VALID[n_idx][k] = False

    def get_label(n, k):
        n_idx = N_IDX[n]
        # n_idx = np.where(N == n)[0][0]
        return (A[n_idx][k], T[n_idx][k], B[n_idx][k], PREV[n_idx][k])

    def dominance_check(a1, t1, b1, a2, t2, b2):
        if a1 <= a2 and t1 <= t2 and b1 >= b2:# and d1 <= d2:
            return True
        else:
            return False

    labels = [(1,2)]
    paths = [[1,2,3]]
    labels.pop()
    paths.pop()

    num_pareto_labels = 0


    # add initial label
    add_label(O, 0, 0, R/2, -3)
    
    # initial queue
    Q = [O]

    found_neg_rc_paths = False

    if preload_paths is not None and use_hrstc == False: # Insert labels of preload paths

        calc_rc = lambda a,t: a + beta - DECAY[max(int(t-sp_time),0)]

        for path in preload_paths:
            a = 0.0
            t = 0.0
            b = R/2
            for i,j in zip(path[:-1], path[1:]):
                i_idx = N_IDX[i]
                j_idx = N_IDX[j]

                new_a = a + round(ALPHA[i],5)

                t_ij = ADJ_MAT[i_idx][j_idx]
                if fst_q:
                    q = fst_q
                else:
                    if use_hrstc:
                        q = CTR[W[i]]
                    else:
                        q = CS[i_idx]

                if b >= t_ij: 
                    if j == D:
                        new_t = t + t_ij + QT + round(q*CT[int(max(b,R/2+t_ij))],3) - round(q*CT[int(b)],3)
                        new_b = max(b, R/2+t_ij) - t_ij
                    else:
                        new_t = t + t_ij
                        new_b = b - t_ij
                else:
                    if j == D:
                        new_t = t + t_ij + QT + round(q*CT[int(R/2+t_ij)],3) - round(q*CT[int(b)],3)
                        new_b = R/2
                    else:
                        new_t = t + t_ij + QT + round(q*CT[int(t_ij)],3) - round(q*CT[int(b)],3)
                        new_b = 0


                # adj_lbl_idx_l = [idx for idx, v in enumerate(VALID[j_idx]) if v == True]

                # if len(adj_lbl_idx_l) == 0:
                #     add_label(j, new_a, new_t, new_b, i)
                #     Q.append(j)
                # else:
                #     dominated = False
                #     for adj_lbl_idx in adj_lbl_idx_l:
                #         adj_a, adj_t, adj_b, adj_prev = get_label(j, adj_lbl_idx)
                #         if dominance_check(adj_a, adj_t, adj_b, new_a, new_t, new_b):
                #             # print('entering dominated', adj_n, new_a, new_t, new_b, new_d, n)
                #             # print('\t entering dominated info',adj_a, adj_t, adj_b, adj_d, new_a, new_t, new_b, new_d)
                #             dominated = True
                #             break
                #     if not dominated:
                #         for adj_lbl_idx in adj_lbl_idx_l:
                #             adj_a, adj_t, adj_b, adj_prev = get_label(j, adj_lbl_idx)
                #             if dominance_check(new_a, new_t, new_b, adj_a, adj_t, adj_b):
                #                 # print('original dominated', adj_n, new_a, new_t, new_b, new_d, n)
                #                 # print('\t original dominated info',new_a, new_t, new_b, new_d, adj_a, adj_t, adj_b, adj_d)
                #                 remove_label(j, adj_lbl_idx)

                #         add_label(j, new_a, new_t, new_b, i)
                #         # print('add label',adj_n, new_a, new_t, new_b, new_d, n)
                #         if j not in Q:
                #             Q.append(j)                

                a = new_a
                t = new_t
                b = new_b


                if j == D:
                    if calc_rc(a,t) <= -0.001:
                        labels.append((a,t))
                        paths.append([i for i in path])
                        found_neg_rc_paths = True

        # print(len(labels), )

        if found_neg_rc_paths:
            return labels, paths



    while len(Q) != 0 and found_neg_rc_paths == False:
        n = Q.pop()
        n_idx = N_IDX[n]
        # n_idx = np.where(N == n)[0][0]

        if max_num_labels > 0:
            if num_pareto_labels >= max_num_labels:
                break


        adj_n_l = ADJ_LIST[n_idx]
        # adj_n_l = [N[idx] for idx,val in enumerate(ADJ_MAT[n_idx]) if val >= 0]


        # not_treated_labels = [idx for idx,l in enumerate(TREAT[n_idx]) if l == False and VALID[n_idx][idx] == True]

        # for i in not_treated_labels:

        for i in range(len(A[n_idx])):
            if TREAT[n_idx][i] == True or VALID[n_idx][i] == False:
                continue

            a, t, b, prev = get_label(n, i)
            TREAT[n_idx][i] = True

            for adj_n in adj_n_l:
                adj_n_idx = N_IDX[adj_n]
                # adj_n_idx = np.where(N == adj_n)[0][0]

                if n == -1:
                    new_a = a
                else:
                    new_a = a + round(ALPHA[n],5)

                t_ij = ADJ_MAT[n_idx][adj_n_idx]

                if fst_q:
                    q = fst_q
                else:
                    if use_hrstc:
                        q = CTR[W[n]]
                    else:
                        q = CS[n_idx]

                if b >= t_ij: #ADJ_MAT[n_idx][adj_n_idx]:
                    if adj_n == D:
                        new_t = t + t_ij + QT + round(q*CT[int(max(b,R/2+t_ij))],3) - round(q*CT[int(b)],3)
                        new_b = max(b, R/2+t_ij) - t_ij
                    else:
                        new_t = t + t_ij
                        new_b = b - t_ij
                else:
                    if adj_n == D:
                        new_t = t + t_ij + QT + round(q*CT[int(R/2+t_ij)],3) - round(q*CT[int(b)],3)
                        new_b = R/2
                    else:
                        new_t = t + t_ij + QT + round(q*CT[int(t_ij)],3) - round(q*CT[int(b)],3)
                        new_b = 0
                
                # new_d = d + ADJ_MAT[n_idx][adj_n_idx]
                prev = n

                adj_lbl_idx_l = [idx for idx, v in enumerate(VALID[adj_n_idx]) if v == True]

                if len(adj_lbl_idx_l) == 0:
                    add_label(adj_n, new_a, new_t, new_b, n)
                    # print('add label',adj_n, new_a, new_t, new_b, new_d, n)
                    Q.append(adj_n)
                else:
                    dominated = False
                    for adj_lbl_idx in adj_lbl_idx_l:
                        adj_a, adj_t, adj_b, adj_prev = get_label(adj_n, adj_lbl_idx)
                        if dominance_check(adj_a, adj_t, adj_b, new_a, new_t, new_b):
                            # print('entering dominated', adj_n, new_a, new_t, new_b, new_d, n)
                            # print('\t entering dominated info',adj_a, adj_t, adj_b, adj_d, new_a, new_t, new_b, new_d)
                            dominated = True
                            break
                    if not dominated:
                        for adj_lbl_idx in adj_lbl_idx_l:
                            adj_a, adj_t, adj_b, adj_prev = get_label(adj_n, adj_lbl_idx)
                            if dominance_check(new_a, new_t, new_b, adj_a, adj_t, adj_b):
                                # print('original dominated', adj_n, new_a, new_t, new_b, new_d, n)
                                # print('\t original dominated info',new_a, new_t, new_b, new_d, adj_a, adj_t, adj_b, adj_d)
                                remove_label(adj_n, adj_lbl_idx)

                        add_label(adj_n, new_a, new_t, new_b, n)
                        # print('add label',adj_n, new_a, new_t, new_b, new_d, n)
                        Q.append(adj_n)


                if adj_n == D:
                    num_pareto_labels = len(adj_lbl_idx_l)


    # D_idx = np.where(N == D)[0][0]
    D_idx = N_IDX[D]
    d_lbl_idx_l = [idx for idx, v in enumerate(VALID[D_idx]) if v == True]
    for d_lbl_idx in d_lbl_idx_l:
        path = [PREV[D_idx][d_lbl_idx],D]
        curr_label_idx = (D_idx, d_lbl_idx)
        broken_path = False
        while path[0] != O:
            i = path[0]
            j = path[1]
            i_idx = N_IDX[i]
            j_idx = N_IDX[j]
            # i_idx = np.where(N == i)[0][0]
            # j_idx = np.where(N == j)[0][0]
            i_lbl_idx_l = [idx for idx, v in enumerate(VALID[i_idx]) if v == True]
            for i_lbl_idx in i_lbl_idx_l:
                a, t, b, prev = get_label(i, i_lbl_idx)

                if i == -1:
                    new_a = a
                else:
                    new_a = a + round(ALPHA[i],5)

                adj_mat_i_j = ADJ_MAT[i_idx][j_idx]

                if fst_q:
                    q = fst_q
                else:
                    if use_hrstc:
                        q = CTR[W[i]]
                    else:
                        q = CS[i_idx]

                if b >= adj_mat_i_j:
                    if j == D:
                        new_t = t + adj_mat_i_j + QT + round(q*CT[int(max(b,R/2+adj_mat_i_j))],3) - round(q*CT[int(b)],3)
                        new_b = max(b, R/2+adj_mat_i_j) - adj_mat_i_j
                    else:
                        new_t = t + adj_mat_i_j
                        new_b = b - adj_mat_i_j
                else:
                    if j == D:
                        new_t = t + adj_mat_i_j + QT + round(q*CT[int(R/2+adj_mat_i_j)],3) - round(q*CT[int(b)],3)
                        new_b = R/2
                    else:
                        new_t = t + adj_mat_i_j + QT + round(q*CT[int(adj_mat_i_j)],3) - round(q*CT[int(b)],3)
                        new_b = 0
                
                # new_d = d + adj_mat_i_j

                if new_a-1e-6 <= A[curr_label_idx[0]][curr_label_idx[1]] <= new_a+1e-6 and new_t-1e-6 <= T[curr_label_idx[0]][curr_label_idx[1]] <= new_t+1e-6 and \
                    new_b-1e-6 <= B[curr_label_idx[0]][curr_label_idx[1]] <= new_b+1e-6:# and new_d-1e-6 <= DIST[curr_label_idx[0]][curr_label_idx[1]] <= new_d+1e-6:
                    path.insert(0, prev)
                    curr_label_idx = (i_idx, i_lbl_idx)
                    break
            else:
                # print('Infinite loop occured!')
                broken_path = True
                break

        if not broken_path:
            labels.append((A[D_idx][d_lbl_idx], T[D_idx][d_lbl_idx]))
            paths.append(path)

    return labels, paths



@numba.njit(cache=True, parallel=False, nogil=True, fastmath=True)
def labeling_numba_preload_path(O, D, N, R, ADJ_MAT, ALPHA, CT, QT, CS, W, CTR, use_hrstc, fst_q, num_preload_paths, DECAY, beta, sp_time, max_num_labels):

    if num_preload_paths > 0 and use_hrstc == False:

        N_IDX = {}
        for (idx,i) in enumerate(N):
            N_IDX[i] = idx

        ADJ = [
            np.array([n_idx for n_idx, val in enumerate(ADJ_MAT[idx]) if val >= 0], dtype=np.int64)
            for idx, i in enumerate(N)
        ]

        source = N_IDX[O]
        target = N_IDX[D]

        n = len(N)

        cost = np.zeros((n,n), dtype=np.float64)

        for i in range(n):
            cost[:, i] = ALPHA[N[i]] if N[i] >= 0 else 0
            
        ksp_paths = ksp(n, ADJ, cost, source, target, K=num_preload_paths)

        preload_paths = [
            np.array([N[i] for i in path], dtype=np.int64)
            for dist, path in ksp_paths
        ]


        labeling_result = labeling_numba(O, D, N, R, ADJ_MAT, ALPHA, CT, QT, CS, W, CTR, use_hrstc, fst_q, preload_paths=preload_paths, DECAY=DECAY, beta=beta, sp_time=sp_time, max_num_labels=max_num_labels)

    else:
        labeling_result = labeling_numba(O, D, N, R, ADJ_MAT, ALPHA, CT, QT, CS, W, CTR, use_hrstc, fst_q, preload_paths=None, DECAY=DECAY, beta=beta, sp_time=sp_time, max_num_labels=max_num_labels)



    return labeling_result


def calculate_reduced_cost(prob, k, label, beta, shortest_time):
    ct = max(int(label[1]-shortest_time[k]),0)
    return label[0] + beta - prob.DECAY[k][ct]

def get_installed_nodes(prob, x):
    return [i for i in prob.N for w in range(len(prob.W)) if sum(x[i*len(prob.W):i*len(prob.W)+w]) > 0.8]

def get_installed_station_types(prob, x):
    return [w for i in prob.N for w in range(len(prob.W)) if sum(x[i*len(prob.W):i*len(prob.W)+w]) > 0.8]

def recalculate_alpha(prob, alpha_k, beta_k, g_star, use_hrstc):
    corrected_alpha_k = {k:{i:{w:0 for w in range(len(prob.W))} for i in prob.N} for k in prob.K}
    for k in prob.K:
        if use_hrstc[k]:
            for i in prob.EN[k]['net'].nodes:
                if i >= 0:
                    for w in range(len(prob.W)):
                        if prob.W[w][1] <= prob.W[g_star[i]][1]:
                            corrected_alpha_k[k][i][w] = alpha_k[k][i]
                        else:
                            corrected_alpha_k[k][i][w] = prob.D_k[k] - beta_k[k]

        else:
            for i in prob.EN[k]['net'].nodes:
                corrected_alpha_k[k][i%len(prob.N)][i//len(prob.N)] = alpha_k[k][i]
    return corrected_alpha_k

def get_corrected_charging_pattern(prob,x):
    g_star = [0]*len(prob.N)
    for i in prob.N:
        installed = False
        for w in range(len(prob.W)):
            if x[i*len(prob.W)+w] > 0.8:
                installed = True
                g_star[i] = w
                break
        if not installed:
            g_star[i] = prob.fst_charger_idx
    return np.array(g_star)

def get_cut(prob, pi_l):
    return cplex.SparsePair(
            ind = ['z'] + [f"x_{i}_{w}" for i in prob.N for w in range(len(prob.W))],
            val = [1] + [-pi_l[i][w] for i in prob.N for w in range(len(prob.W))]
            )

def get_cut_k(prob, pi_l_k, k):
    return cplex.SparsePair(
            ind = [f'z_{k}'] + [f"x_{i}_{w}" for i in prob.N for w in range(len(prob.W))],
            val = [1] + [-pi_l_k[i][w] for i in prob.N for w in range(len(prob.W))]
            )

def get_pi(prob, alpha_k):
    pi = {i:{w:sum([alpha_k[k][i][w] for k in prob.K]) for w in range(len(prob.W))} for i in prob.N}
    return pi

def get_pi_k(prob, alpha_k, k):
    pi_k = {i:{w:alpha_k[k][i][w] for w in range(len(prob.W))} for i in prob.N}
    return pi_k

def get_covered_paths(model):
    x = [model.master.solution.get_values(f"x_{i}_{w}") for i in model.prob.N for w in range(len(model.prob.W))]
    g_star = get_corrected_charging_pattern(model.prob, x)

    for k in model.prob.K:
        if model.prob.use_BnC:
            update_sub_model(model.prob, k, x, g_star, model.sub_models, model.cb.paths, model.prob.shortest_time, model.prob.use_hrstc[k])
        else:
            update_sub_model(model.prob, k, x, g_star, model.sub_models, model.paths, model.prob.shortest_time, model.prob.use_hrstc[k])
        model.sub_models[k].solve()

    covered_EN_paths_k = {}
    covered_full_paths_k = {}
    for k in model.prob.K:
        if model.prob.use_BnC:
            covered_paths = [path for p_idx,path in enumerate(model.cb.paths[k]) if model.cb.sub_models[k].solution.get_dual_values(f'const_{p_idx}') > 0.8]
        else:
            covered_paths = [path for p_idx,path in enumerate(model.paths[k]) if model.sub_models[k].solution.get_dual_values(f'const_{p_idx}') > 0.8]

        if covered_paths:
            if len(covered_paths) > 1:
                print('multiple covered path for one OD pair!')
            if model.prob.use_hrstc[k]:
                P = covered_paths[0]
                G = [g_star[i] if i >= 0 else 0 for i in P]
                f = get_flow(model.prob, k, P, G, get_time=False, shortest_time=model.prob.shortest_time)
                covered_EN_paths_k[k] = (P, G, f)
                full_path, length = get_full_path(model.prob, k, P)
                time = float(get_flow(model.prob, k, P, G, get_time=True))
                covered_full_paths_k[k] = (full_path, length, time, f)
            else:
                P,G,f = covered_paths[0]
                covered_EN_paths_k[k] = (P, G, f)
                full_path, length = get_full_path(model.prob, k, P)
                time = float(get_flow(model.prob, k, P, G, get_time=True))
                covered_full_paths_k[k] = (full_path, length, time, f)
    return covered_EN_paths_k, covered_full_paths_k

def get_full_path(prob, k, P):
    full_path = [prob.K[k][0]]
    
    prev_i = prob.K[k][0]
    length = 0
    for i in P[1:-1]:
        full_path.extend(list(nx.dijkstra_path(prob.G, prev_i, i, weight='length'))[1:])
        # length += nx.dijkstra_path_length(prob.G, prev_i, i, weight='length')
        length += prob.G_shortest_matrix[prev_i][i]
        prev_i = i

    if full_path[-1] != prob.K[k][1]:
        # length += nx.dijkstra_path_length(prob.G, full_path[-1], prob.K[k][1], weight='length')
        length += prob.G_shortest_matrix[full_path[-1]][prob.K[k][1]]
        full_path.append(prob.K[k][1])

    return full_path, length

def get_path_length(prob, p):
    length = 0
    for i_idx, i in enumerate(p[1:-2]):
        length += prob.G_shortest_matrix[i][p[1:-1][i_idx+1]]
    return length


class LazyCutGenCallback(cplex.callbacks.LazyConstraintCallback):
    def __call__(self):
        if time.time() - self.start_time > self.time_limit:
            self.abort()
        else:
            self.num_called += 1

            g_star = get_corrected_charging_pattern(self.prob, self.get_values())
            
            alpha_k, beta_k, cut_gen_time, labeling_time, lp_solve_time = cut_generation(self.prob, self.get_values(), self.sub_models, self.paths, self.shortest_time, self.use_hrstc, self.start_time, self.time_limit, self.thread_executor)

            self.lp_time += lp_solve_time
            self.cut_gen_time += cut_gen_time
            self.labeling_time += labeling_time

            corrected_alpha_k = recalculate_alpha(self.prob, alpha_k, beta_k, g_star, self.use_hrstc)

            cut_k = {}
            for k in self.prob.K:
                pi_l_k = get_pi_k(self.prob, corrected_alpha_k, k)

                cut = get_cut_k(self.prob, pi_l_k, k)
                self.generated_cuts.append((cut, beta_k[k], self.get_values()))
                cut_k[k] = cut

            for k, cut in cut_k.items():
                self.add(cut, 'L', beta_k[k])

                self.num_cuts += 1