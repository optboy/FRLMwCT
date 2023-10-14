from itertools import combinations
import numpy as np
import math
import networkx as nx
from . import benders

class Problem:
    def __init__(self, net, K, D_k, G_shortest_matrix, R=15, L=1.3, B=30, QT=0, W=[(1,1),(0.5,3)], fst_q=0.5, alpha=1, beta=0.1, theta=0.8, delta=1, tau=0.5, decay_f='n', use_CT=True, use_BnC=True, num_preload_paths=10, max_num_labels=-1, use_length_limit=False):
        self.net = net
        self.G = self.net.G
        self.N = self.G.nodes
        self.A = self.G.edges
        self.K = K
        self.R = R
        self.L = L
        self.B = B
        self.QT = QT
        self.W = W
        self.fst_q = fst_q
        self.Q = [q for q,c in W]
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.decay_f = decay_f
        self.C_i = {i:[c for q,c in W] for i in self.G}
        self.D_k = D_k
        self.G_shortest_matrix = G_shortest_matrix

        self.K_idx = {v:k for k,v in self.K.items()}

        self.weighted_nodes = set([i for v in K.values() for i in v])
        
        # self.fwd_K = {self.K_idx[v]:v for v in [v for v in combinations(self.weighted_nodes, r=2)]}
        self.fwd_K = {self.K_idx[v]:v for v in self.K.values()}
        self.bwd_K = {self.K_idx[(v2,v1)]:(v2,v1) for k,(v1,v2) in self.fwd_K.items()}

        fwd_D_k = {k:self.D_k[k] for k in self.fwd_K}
        demand_rank = [k for k,v in sorted(fwd_D_k.items(), key=lambda item: item[1], reverse=True)]
        high_demand_k = demand_rank[:int(len(self.fwd_K)*tau)]
        high_demand_k = high_demand_k + [self.K_idx[tuple([v for v in reversed(self.K[k])])] for k in high_demand_k]
        self.use_hrstc = np.array([False if k in high_demand_k else True for k in self.K])
        self.use_BnC = use_BnC
        self.num_preload_paths = num_preload_paths
        self.max_num_labels = max_num_labels
        self.use_length_limit = use_length_limit

        self.EN = {idx:self.net.make_expanded_network(self.G, i, j, L, R, G_shortest_matrix, W) if idx in high_demand_k else
                    self.net.make_expanded_network(self.G, i, j, L, R, G_shortest_matrix) for idx,(i,j) in K.items()}

        # self.EN

        self.fst_charger_idx = np.argmin(self.Q)

        if use_CT:
            battery_model = BatteryModel(self.alpha, self.R, self.theta)
            self.RT = lambda x: battery_model.rt_by_dist(x)
            self.CT = np.array([self.RT(i) for i in range(R+1)])
        else:
            self.CT = np.array([0 for i in range(R+1)])

        self.EN_INFO = []
        self.shortest_time = {}
        self.shortest_time_path = {}
        self.shortest_time_full_path = {}
        self.DECAY = []
        for k in self.K:
            G = self.EN[k]['net']
            O, D = -1, -2
            N = np.array(G.nodes)
            ADJ_MAT = [[-1]*len(G.nodes) for i in range(len(G.nodes))]
            for n1, val1 in G.adjacency():
                for n2, val2 in val1.items():
                    ADJ_MAT[np.where(N == n1)[0][0]][np.where(N == n2)[0][0]] = val2['length']
            ADJ_MAT = np.array(ADJ_MAT)

            CT = self.CT
            QT = self.QT

            if k in high_demand_k:
                CS = np.array([G.nodes[i]['q'] for i in G.nodes])
            else:
                CS = np.array([])

            Q = np.array(self.Q)

            #shortest path
            ALPHA = np.array([0 for i in self.N for w in W])
            W = np.array([np.argmin([w[0] for w in self.W])]*len(self.N)*len(self.W))
            labels, paths = benders.labeling_numba(O, D, N, R, ADJ_MAT, ALPHA, CT, 0, CS, W, Q, use_hrstc=self.use_hrstc[k], fst_q=fst_q)
            self.shortest_time[k] = min([t for a,t in labels])
            if self.use_hrstc[k]:
                P = paths[np.argmin([t for a,t in labels])]
                self.shortest_time_path[k] = (P,self.D_k[k])
                self.shortest_time_full_path[k] = benders.get_full_path(self, k, P)
            else:
                path = paths[np.argmin([t for a,t in labels])]
                P,G = benders.get_PnG_from_sol(self, path)
                self.shortest_time_path[k] = (P,self.D_k[k])
                self.shortest_time_full_path[k] = benders.get_full_path(self, k, P)

            # Decay
            X = range(int(self.shortest_time[k]), int(self.shortest_time[k]*30))
            cost=[]
            for i in X:
                if decay_f == 'n':
                    cost.append(min(self.D_k[k],round(self.D_k[k]*delta*math.exp((-beta*((i-self.shortest_time[k])/self.shortest_time[k]))))))
                elif decay_f == 'l':
                    cost.append(round(max(self.D_k[k]*(1-((i-self.shortest_time[k])/(beta*self.shortest_time[k]))),0)))
                elif decay_f == 's':
                    # max(1 / (1 + alpha * math.exp((beta * DD - dq)/(len(DD_l)/10))),0)
                    cost.append(round(max( self.D_k[k] * (1 / (1 + delta * math.exp( (beta * (i - self.shortest_time[k]) - self.shortest_time[k]*0.5) / (self.shortest_time[k]/10) ) )), 0)))
                    # cost.append(round(self.D_k[k]*(1/(1+delta*math.exp(((10*beta/(self.shortest_time[k]))*(i-self.shortest_time[k] - self.shortest_time[k]*0.2)))))))
                elif decay_f == 'e':
                    # max(1 - (alpha * math.exp(beta/len(DD_l)*(DD-len(DD_l)))),0)
                    # 1 - (delta * math.exp*(beta/len(X)*(i - self.shortest_time[k] - len(X))))
                    cost.append(round(max(self.D_k[k] * (1 - (delta * math.exp(beta/self.shortest_time[k] * (i - self.shortest_time[k] - self.shortest_time[k])))), 0)))

            self.DECAY.append(cost)
            DECAY = np.array(self.DECAY[k])
            DECAY = DECAY + (self.D_k[k] - DECAY[0])

            self.EN_INFO.append((O, D, N, R, np.array(ADJ_MAT), DECAY, CT, QT, CS, Q))

class BatteryModel:
    def __init__(self, alpha=1, B=500, theta=0.8, linear_mode=False):
    
        self.alpha = alpha  # slofe for CC-stage
        self.B = B

        if linear_mode:
            theta = 1.0
        self.theta = theta

        tt = range(0, B*10)

        def it(t, b): 
            if theta < 1:
                if b <= theta*B:
                    return alpha
                else:
                    return alpha / (1-theta) * (B+1-b)/B
            else:
                return alpha

        soc = []
        i = []
        b = 0
        for t in tt:
            soc.append(b)
            db = it(t, b)
            i.append(db)
            b += db


        b = 0
        t = 0
        ct = []
        while b <= B:
            while soc[t] < b:
                t += 1
            if soc[t] == b:
                ct.append(t)
            else:
                t_1 = ct[-1]
                #ct.append(int(round(lmd*t + (1-lmd)*t_1)))
                ct.append(t)
            b += 1

        self.I = i # current over time
        self.soc = soc # state of charge over time
        self.ct = np.array(ct) # recharge timr by distance


    def rt_by_dist(self, d):
        return self.ct[np.minimum(np.round(d).astype(np.int64), self.B)]