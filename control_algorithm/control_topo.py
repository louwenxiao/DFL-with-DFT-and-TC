import sys
import time
import numpy as np
from pulp import *

def calculate_time(links, compression_rates, bandwidth, comp_time, model_size):
    nodes_num = len(compression_rates)
    link_speed = np.zeros_like(links, dtype=np.float)
    link_time = np.zeros_like(links, dtype=np.float)

    # print(bandwidth)
    for i in range(nodes_num):
        for j in range(nodes_num):
            if links[i][j] == 0 or i == j:
                link_time[i][j] = 0
            elif links[i][j] == 1:
                link_speed[i][j] = np.min([bandwidth[0][i] / np.sum(links[i]), bandwidth[1][j] / np.sum(links[j])])
                link_time[i][j] = compression_rates[i] * model_size / link_speed[i][j] + comp_time[i]
    # print("link speed\n", link_speed)
    # print(link_time)

    return link_time

def calculate_distance(curr_distance, links, compression_rates):
    nodes_num = len(compression_rates)
    next_distance = np.zeros_like(links, dtype=np.float).tolist()
    for i in range(nodes_num):
        for j in range(nodes_num):
            if i == j:
                continue
            next_distance[i][j] = (1 - links[i][j] * compression_rates[j] ) * curr_distance[i][j]
    # print(next_distance)
    # avg_distance = np.sum(next_distance) / (nodes_num * nodes_num)
    # print(avg_distance)
    return next_distance

def optimize_rates(links, curr_distance, bandwidth, distance_target, comp_time, model_size, lower=0.001):
    nodes_num = len(links)
    prob = LpProblem('minimizeTime', LpMinimize)
    minimize_object = LpVariable('upper_bound')
    prob += minimize_object

    rates = list()
    for node in range(nodes_num):
        rates.append(LpVariable("node_"+str(node), lower, 1))

    # print("distance", curr_distance)
    # print("links", links)
    # print("rates", rates)
    # print("comp ", comp_time)
    distance = np.sum(calculate_distance(curr_distance, links, rates)) / (nodes_num * nodes_num)
    # print(distance)
    # print("distance target:", np.average(curr_distance), distance_target)
    prob += distance <= distance_target
    # link_time = list()
    for i in range(nodes_num):
        for j in range(nodes_num):
            if links[i][j] == 1:
                link_speed = np.min([bandwidth[0][i] / np.sum(links[i]), bandwidth[1][j] / np.sum(links[j])])
                link_time = rates[i] * model_size / link_speed + comp_time[i]
                prob +=  link_time <= minimize_object
    prob.solve(PULP_CBC_CMD(msg=0))
    # print("ratio results")
    for node in range(nodes_num):
        rates[node] = value(rates[node])

        if rates[node] < lower:
            rates[node] = lower
        if rates[node] > 1.0:
            rates[node] = 1.0
    
    return rates

def dfs_topo(topo, visited, node):
    visited[node] = True

    for i in range(len(topo)):
        if topo[node][i] == 1 and visited[i] == False:
            dfs_topo(topo, visited, i)

def connected_topo(topo):
    visited = [False for _ in range(len(topo))]
    dfs_topo(topo, visited, 0)

    for node in range(len(topo)):
        if visited[node] == False:
            # print("Not connected!")
            # print(topo)
            return False
    return True
def find_largest_k(matrix, k):
    shape = matrix.shape

    ret = list()
    k_idxes = np.argpartition(matrix.ravel(), -k)[-k:]
    for i in k_idxes:
        ret.append(np.unravel_index(i, shape))
    
    return np.array(ret)

def construct_topo(mode, consensus_distance, bandwidth, target, comp_time, model_size, lower, default_topo=None):
    nodes_num = len(consensus_distance)
    if mode == "rate-adaptive":
        optimal_rates = optimize_rates(default_topo, consensus_distance, bandwidth, target, comp_time, model_size, lower)
        return default_topo, optimal_rates
    if mode == "topo-adaptive":
        optimal_rates = np.array([lower for _ in range(nodes_num)])
    optimal_topo = np.ones_like(consensus_distance, dtype=np.int)
    for node_idx in range(nodes_num):
        optimal_topo[node_idx][node_idx] = 0
    if mode == "adaptive":
        optimal_rates = optimize_rates(optimal_topo, consensus_distance, bandwidth, target, comp_time, model_size, lower)
    
    minimum_time = np.max(calculate_time(optimal_topo, optimal_rates, bandwidth, comp_time, model_size))

    base_rates = optimal_rates.copy()
    base_topo = optimal_topo.copy()
    top_k = nodes_num
    while np.sum(base_topo) > (nodes_num - 1) * 2:
        curr_link_distance = np.array(calculate_distance(consensus_distance, base_topo, base_rates))
        curr_link_time = calculate_time(base_topo, base_rates, bandwidth, comp_time, model_size)

        for x in range(nodes_num):
            for y in range(x+1, nodes_num):
                curr_link_time[x][y] = (curr_link_time[x][y] + curr_link_time[y][x]) / 2
                curr_link_time[y][x] = 0

        bottleneck_links = find_largest_k(curr_link_time, top_k)

        curr_topo = base_topo.copy()
        for x, y in bottleneck_links:
            curr_topo[x][y] = 0
            curr_topo[y][x] = 0
            if connected_topo(curr_topo) == False:
                curr_topo[x][y] = 1
                curr_topo[y][x] = 1
        if mode == "adaptive":
            curr_rates = optimize_rates(curr_topo, consensus_distance, bandwidth, target, comp_time, model_size, lower)
        else:
            curr_rates = optimal_rates
        curr_time = np.max(calculate_time(curr_topo, curr_rates, bandwidth, comp_time, model_size))
        curr_distance = np.average(calculate_distance(consensus_distance, curr_topo, curr_rates))

        if curr_time < minimum_time and curr_distance <= target+0.1:
            base_topo = curr_topo.copy()
            optimal_topo = curr_topo.copy()
            optimal_rates = curr_rates.copy()
            minimum_time = curr_time
            top_k = int(np.sqrt(np.sum(base_topo)))
        else:
            top_k = max(int(top_k / 2), 1)
            if top_k == 1:
                i, j = np.unravel_index(np.argmax(curr_link_time, axis=None), curr_link_time.shape)
                find = False
                while curr_link_time[i][j] > 0:
                    # print("search ", i, " ", j)
                    base_topo[i][j] = 0
                    base_topo[j][i] = 0
                    if connected_topo(base_topo):
                        if mode == "adaptive":
                            curr_rates = optimize_rates(base_topo, consensus_distance, bandwidth, target, comp_time, model_size, lower)
                        else:
                            curr_rates = optimal_rates
                        curr_time = np.max(calculate_time(base_topo, curr_rates, bandwidth, comp_time, model_size))
                        curr_distance = np.average(calculate_distance(consensus_distance, base_topo, curr_rates))

                        if curr_time < minimum_time and curr_distance <= target+0.1:
                            optimal_topo = base_topo.copy()
                            optimal_rates = curr_rates.copy()
                            minimum_time = curr_time
                            find = True
                            break
                    base_topo[i][j] = 1
                    base_topo[j][i] = 1
                    curr_link_time[i][j] = 0
                    i, j = np.unravel_index(np.argmax(curr_link_time, axis=None), curr_link_time.shape)
                if find == False:
                    break
                else:
                    top_k = int(np.sqrt(np.sum(base_topo)))

    return optimal_topo, optimal_rates
