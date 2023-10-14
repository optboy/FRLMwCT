import cplex
import networkx as nx
import itertools

def path_based_FRLM(model):
    paths = {k:list(nx.all_simple_paths(model.prob.EN[k]['net'],model.prob.K[k][0], model.prob.K[k][1], cutoff=8)) for k in model.prob.K}

    paths = {}
    for k in model.prob.K:
        path_q = []
        paths_k = list(nx.all_simple_paths(model.prob.EN[k]['net'],model.prob.K[k][0], model.prob.K[k][1], cutoff=8))
        for path in paths_k:
            w = list(itertools.product(*[model.prob.W for p in path]))
            path_q.append((path, w))
        paths[k] = path_q

    P_k = {k:range(len(v)) for k,v in paths.items()}

    cost = {k:{} for k in model.prob.K}

    for k, path in paths.items():
        for path_idx, (nodes, cst) in enumerate(path):
            cost[k][path_idx] = model.get_flow(k,nodes,cst)


    cpx = cplex.Cplex()

    # variables
    cpx.variables.add(names=[f"x_{i}_{w}" for i in model.prob.N for w in range(len(model.prob.W))],
                    types=['B' for i in model.prob.N for w in range(len(model.prob.W))])

    cpx.variables.add(names=[f"y_{k}_{p}" for k in model.prob.K for p in P_k[k]],
                    types=['B' for k in model.prob.K for p in P_k[k]])

    # constraints
    params = []
    for k in model.prob.K:
        for i in model.prob.N:
            for w in range(len(model.prob.W)):
                for p in P_k[k]:
                    path,csts = paths[k][p]
                    if i in path:
                        if w == csts[path.index(i)]:
                            params.append((k,i,w,p))

    cpx.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(
                ind = [f"x_{i}_{w}"] + [f"y_{k}_{p}"],
                val = [1] + [-1]
            ) for k,i,w,p in params
        ],
        senses=['G' for k,i,w,p in params],
        rhs=[0 for k,i,w,p in params],
        names=[f'station_installation_{k}_{i}_{w}_{p}' for k,i,w,p in params]
    )

    cpx.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(
                ind = [f"x_{i}_{w}" for w in range(len(model.prob.W))],
                val = [1 for w in range(len(model.prob.W))]
            ) for i in model.prob.N
        ],
        senses=['L' for i in model.prob.N],
        rhs=[1 for i in model.prob.N],
        names=[f'unique_station_type_{i}' for i in model.prob.N]
    )

    cpx.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(
                ind = [f"x_{i}_{w}" for i in model.prob.N for w in range(len(model.prob.W))],
                val = [model.prob.C_i[i][w] for i in model.prob.N for w in range(len(model.prob.W))]
            )
        ],
        senses=['L'],
        rhs=[model.prob.B],
        names=[f'Budget']
    )         

    cpx.objective.set_sense(cpx.objective.sense.maximize)

    cpx.objective.set_linear([(f"y_{k}_{p}", cost[k][p]) for k in model.prob.K for p in P_k[k]])

    cpx.solve()

    return cpx