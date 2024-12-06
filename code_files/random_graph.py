import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import random

def check_connectivity(graph_desc, start, end):
    if isinstance(graph_desc, dict):
        G = nx.Graph(graph_desc)
        return nx.has_path(G, start, end)

    else:
        G = nx.from_numpy_matrix(graph_desc)
        return nx.has_path(G, start, end)


def conn_prob(adj_prob_matx, start, end, T = 1000):
    n = adj_prob_matx.shape[0]
    connected = []
    for _ in range(T):
        adj_list = {}
        for i in range(n):
            adj_list[i] = []
            for j in range(i + 1, n):
                if np.random.binomial(1, adj_prob_matx[i,j]) == 1:
                    adj_list[i].append(j)

        # print(adj_list)
        connected.append(int(check_connectivity(adj_list, start, end)))

    # print(connected)
    return np.mean(connected), np.std(connected)

def conn_prob_grid_1_iter(args):
    dim, hor_edge, vert_edge, start, end = args
    adj_list = {}
    for i in range(dim):
        for j in range(dim):
            index = i * dim + j
            adj_list[index] = []
            if random.random() < hor_edge[i, j]:
                adj_list[index].append(index + 1)

            if random.random() < vert_edge[i, j]:
                adj_list[index].append(index + dim)

    return int(check_connectivity(adj_list, start, end))


def conn_prob_grid(dim, hor_edge, vert_edge, start, end, T = 1000):
    #grid wraps around
    assert hor_edge.shape == (dim, dim)
    assert vert_edge.shape == (dim, dim)

    args = [(dim, hor_edge, vert_edge, start, end) for _ in range(T)]

    with ProcessPoolExecutor() as executor:
        connected = list(executor.map(conn_prob_grid_1_iter, args))
    

    # connected = []
    # for it_num in range(T):
    #     if it_num%100 == 0:
    #         print("it_num:", it_num)
        # adj_list = {}
        # for i in range(dim):
        #     for j in range(dim):
        #         index = i * dim + j
        #         adj_list[index] = []
        #         if j < dim - 1 and np.random.binomial(1, hor_edge[i,j]) == 1:
        #             adj_list[index].append(index + 1)

        #         if i < dim - 1 and np.random.binomial(1, vert_edge[i,j]) == 1:
        #             adj_list[index].append(index + dim)

        # connected.append(int(check_connectivity(adj_list, start, end)))

    p_hat = np.mean(connected)
    return p_hat, 1/T*p_hat*(1-p_hat)


if __name__ == "__main__":
    # adj_prob_matx = np.array([
    #     [0, 0.5, 0.5, 0],
    #     [0.5, 0, 0, 0.5],
    #     [0.5, 0, 0, 0.5],
    #     [0, 0.5, 0.5, 0]])

    # print(conn_prob(adj_prob_matx, 0, 3, 10000))

    # hor_grid = np.array([
    #     [0.5],
    #     [0.5]])

    # ver_grid = np.array([
    #     [0.5, 0.5]])

    # print(conn_prob_grid(2, hor_grid, ver_grid, 0, 3, 10000))

    # dim = 100
    # hor_grid = 0.5*np.ones((dim, dim))
    # ver_grid = 0.5*np.ones((dim, dim))

    # print(conn_prob_grid(dim, hor_grid, ver_grid, 0, dim**2 - 1, 10000))

    dim = 10
    ps = np.linspace(0,0.45,10)
    p_mat = (1-2*ps)**2
    print(p_mat)


    for p in p_mat:
        hor_grid = p*np.ones((dim, dim))
        ver_grid = p*np.ones((dim, dim))


        print(conn_prob_grid(dim, hor_grid, ver_grid, 0, dim/2*(1+dim), 100000))







