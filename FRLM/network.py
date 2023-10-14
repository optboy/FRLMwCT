import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from itertools import product

class Net:
    def __init__(self, dataDir, dist_multiplier=1, allow_backward=True, middle_node_sep_len=-1):
        self.dataDir = dataDir
        self.nodes = [n.split() for n in open(self.dataDir + '/NODES.txt', 'r')]
        self.arcs = [[round(float(v)) for v in a.split()] for a in open(self.dataDir + '/ARCS.txt', 'r')]
        self.num_nodes = len(self.nodes)
        self.dist_multiplier = dist_multiplier
        self.allow_backward = allow_backward
        
        self.G = nx.Graph()

        for n in self.nodes:
            self.G.add_node(int(n[0])-1, weight=int(n[1]), pos=(float(n[2]),float(n[3])))

        for a in self.arcs:
            i, j = a[2]-1, a[3]-1
            self.G.add_edge(i, j, dist=int(round(math.sqrt((self.G.nodes[i]['pos'][0] - self.G.nodes[j]['pos'][0])**2 + (self.G.nodes[i]['pos'][1] - self.G.nodes[j]['pos'][1])**2)*dist_multiplier, 0)),
                                  length=a[1]*dist_multiplier, weight=a[1])
            # self.G.add_edge(i, j, length=int(round(math.sqrt((self.G.nodes[i]['pos'][0] - self.G.nodes[j]['pos'][0])**2 + (self.G.nodes[i]['pos'][1] - self.G.nodes[j]['pos'][1])**2)*dist_multiplier, 0)))
            # self.G.add_edge(i, j, length=a[1]*dist_multiplier)
            # self.G.add_edge(i, j, weight=a[1])

        if middle_node_sep_len > 0:
            self.make_middle_nodes(self.G, middle_node_sep_len)

    def draw_network(self, G, curved=False):
        plt.figure(figsize=(10,8))
        pos = {n:(G.nodes[n]['pos'][0],G.nodes[n]['pos'][1]) for n in G.nodes()}
        if curved:
            nx.draw_networkx(G, pos=pos, connectionstyle="arc3,rad=0.3")
        else:
            nx.draw_networkx(G, pos=pos, node_size=3)
    
    def make_expanded_network(self, G_orig, i, j, L, R, G_shortest_matrix, W=None):
        sp_len = G_shortest_matrix[i][j] #nx.dijkstra_path_length(G_orig, i, j, weight='length')

        lengths, paths = nx.single_source_dijkstra(G_orig, i, weight='length')
        EN = nx.DiGraph()

        EN.add_node(-1, pos=G_orig.nodes[i]['pos'], q=10e8, c=10e8)

        EN.add_node(-2, pos=G_orig.nodes[j]['pos'], q=10e8, c=10e8)

        if W is not None:
            range_W = range(len(W))

        selected_nodes = []
        for k in G_orig.nodes:
            detour_len = G_shortest_matrix[i][k] + G_shortest_matrix[k][j] #nx.dijkstra_path_length(G_orig, i, k, weight='length') + nx.dijkstra_path_length(G_orig, k, j, weight='length')
            if detour_len <= sp_len * L:
                selected_nodes.append(k)
                if W is not None:
                    for w in range_W:
                        EN.add_node(k+w*len(G_orig.nodes), pos=G_orig.nodes[k]['pos'], q=W[w][0], c=W[w][1])
                else:
                    EN.add_node(k, pos=G_orig.nodes[k]['pos'])

        for k in selected_nodes:
            s_to_k_len = G_shortest_matrix[i][k] #nx.dijkstra_path_length(G_orig, i, k, weight='length')
            if s_to_k_len <= R/2:
                if W is not None:
                    for w in range_W:
                        EN.add_edge(-1, k+w*len(G_orig.nodes), length=s_to_k_len)
                else:
                    EN.add_edge(-1, k, length=s_to_k_len)
            
            k_to_t_len = G_shortest_matrix[k][j] #nx.dijkstra_path_length(G_orig, k, j, weight='length')
            if k_to_t_len <= R/2:
                if W is not None:
                    for w in range_W:
                        EN.add_edge(k+w*len(G_orig.nodes), -2, length=k_to_t_len)
                else:
                    EN.add_edge(k, -2, length=k_to_t_len)

        for p in selected_nodes:
            for q in selected_nodes:
                if p != q:
                    if self.allow_backward or lengths[p] <= lengths[q]: # Prevent revrese arcs
                        ij_len = G_shortest_matrix[p][q] #nx.dijkstra_path_length(G_orig, p, q, weight='length')
                        if ij_len <= R:
                            if W is not None:
                                for w in range_W:
                                    for adj_w in range_W:
                                        EN.add_edge(p+w*len(G_orig.nodes), q+adj_w*len(G_orig.nodes), length=ij_len)
                            else:
                                EN.add_edge(p, q, length=ij_len)
        
        return {'net':EN, 'sp':sp_len}

    def make_demand(self, OD_set):
        D_k = {idx:int((self.G.nodes[i]['weight']*self.G.nodes[j]['weight']) / nx.dijkstra_path_length(self.G,i,j,weight='weight')**1.5)
            for idx,(i,j) in OD_set.items()}
        return D_k

    def get_pos(self,n1,n2,sep_len):
        x = [n1[0], n2[0]]
        y = [n1[1], n2[1]]
        degree = np.arctan2(y[1] - y[0], x[1] - x[0])
        dx = math.cos(degree)*(sep_len)
        dy = math.sin(degree)*(sep_len)
        return (x[0]+dx, y[0]+dy)

    def make_middle_nodes(self, G, sep_len):
        num_node = len(G.nodes)
        for e in list(G.edges):
            edge_len = G[e[0]][e[1]]['weight']
            length = G[e[0]][e[1]]['length']
            n = math.ceil(edge_len / sep_len)-1
            if n > 0:
                prev_node = e[0]
                for order in range(n):
                    G.add_node(num_node, pos=self.get_pos(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos'],(sep_len)*((length/self.dist_multiplier)/edge_len)*(order+1)), weight=0)
                    G.add_edge(prev_node, num_node,length=math.floor(sep_len*(length/edge_len)))
                    prev_node = num_node
                    num_node += 1
                G.add_edge(num_node-1, e[1],length=math.floor(length-n*sep_len*(length/edge_len)))
                G.remove_edge(*e)

    def get_K(self, num_node, L, R, G_shortest_matrix):
        node_weight_l = [[i,self.G.nodes[i]['weight']] for i in self.G.nodes if self.G.nodes[i]['weight'] > 0]
        weighted_nodes = [i for i,w in sorted(node_weight_l,key=lambda l:l[1], reverse=True)][:num_node]
        OD_pairs = [v for v in product(weighted_nodes, repeat=2) if v[0] != v[1]]
        possible_K = {k:v for k,v in enumerate(OD_pairs)}
        possible_D_k = self.make_demand(possible_K)

        EN = {idx:self.make_expanded_network(self.G, i, j, L, R*self.dist_multiplier, G_shortest_matrix) for idx,(i,j) in possible_K.items()}
        K = {}
        D_k = {}
        for k,v in possible_K.items():
            if nx.has_path(EN[k]['net'], -1, -2):
                K[len(K)] = v
                D_k[len(D_k)] = possible_D_k[k]

        return K, D_k