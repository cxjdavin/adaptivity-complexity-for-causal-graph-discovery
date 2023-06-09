from causaldag import DAG
import random
import networkx as nx
import numpy as np

from collections import defaultdict
from itertools import combinations
from networkx.algorithms import chordal_graph_cliques

import math
import sys
sys.path.insert(0, './PADS')
import Chordal

'''
Get (directed) adjacency list from networkx graph
'''
def get_adj_list(nx_G):
    adj_list = dict()
    for node, nbrdict in nx_G.adjacency():
        adj_list[node] = [nbr for nbr in nbrdict.keys()]
    return adj_list

'''
Compute clique tree of a chordal graph
'''
def compute_clique_tree(nx_G):
    assert nx.is_chordal(nx_G)

    # Compute clique graph H
    # Each node is a maximal clique of G
    # Each edge has weight equal to size of intersection between two maximal cliques
    maximal_cliques = list(chordal_graph_cliques(nx_G))
    H = nx.Graph()
    for i in range(len(maximal_cliques)):
        H.add_node(i, mc=maximal_cliques[i])
    for u,v in combinations(H.nodes(), 2):
        wt = len(H.nodes[u]['mc'].intersection(H.nodes[v]['mc']))
        H.add_edge(u, v, weight=wt)

    # Compute clique tree by taking maximum spanning tree of H
    T = nx.maximum_spanning_tree(H)

    return T

'''
Compute maximum independent set via PEO in O(n + m) time.

References:
Fănică Gavril. Algorithms for minimum coloring, maximum clique, minimum covering by cliques, and maximum independent set of a chordal graph. SIAM Journal on Computing, 1972
Jospeh Y-T Leung. Fast algorithms for generating all maximal independent sets of interval, circular-arc and chordal graphs. Journal of Algorithms, 1984
'''
def maximum_independent_set(nx_G):
    assert nx.is_chordal(nx_G)

    adj_list = get_adj_list(nx_G)
    G = dict()
    for v in nx_G.nodes():
        G[v] = adj_list[v]
    peo = Chordal.PerfectEliminationOrdering(G)

    # Sanity check: For every vertex v, v and its later neighbors form a clique
    assert len(peo) == nx_G.number_of_nodes()
    for idx in range(len(peo)):
        v = peo[idx]
        later_nbrs = []
        for later_idx in range(idx+1, len(peo)):
            w = peo[later_idx]
            if w in adj_list[v]:
                later_nbrs.append(w)
        for u in later_nbrs:
            for w in later_nbrs:
                if u != w:
                    assert u in adj_list[w] and w in adj_list[u]

    actual_to_peo = dict()
    for idx in range(len(peo)):
        actual_to_peo[peo[idx]] = idx

    S = set()
    for idx in range(len(peo)):
        v = peo[idx]
        no_earlier_nbr_in_S = True
        for w in adj_list[v]:
            if actual_to_peo[w] < idx and w in S:
                no_earlier_nbr_in_S = False
                break
        if no_earlier_nbr_in_S:
            S.add(v)

    # Sanity check: S is an independent set
    for u in S:
        for v in S:
            if u == v:
                continue
            else:
                assert u not in adj_list[v]
                assert v not in adj_list[u]

    # Sanity check: S is a maximal independent set
    for v in nx_G.nodes():
        if v not in S:
            has_neighbor_in_S = False
            for w in adj_list[v]:
                if w in S:
                    has_neighbor_in_S = True
                    break
            assert has_neighbor_in_S

    return S

'''
Compute minimum vertex cover by removing maximum independent set from vertex set.
'''
def minimum_vertex_cover(nx_G):
    assert nx.is_chordal(nx_G)

    maxIS = maximum_independent_set(nx_G)
    minVC = set(nx_G.nodes()).difference(maxIS)

    # Sanity check: minVC is vertex cover
    for u,v in nx_G.edges():
        assert u in minVC or v in minVC

    # Sanity check: minVC is minimal
    required = set()
    for u,v in nx_G.edges():
        if u in minVC and v not in minVC:
            required.add(u)
        if u not in minVC and v in minVC:
            required.add(v)
    assert len(required) == len(minVC)

    return minVC

'''
Refactored from [CSB22]

The following few lines are copied from the proof of Lemma 1
Let n = p_d a^d + r_d and n = p_{d-1} a^{d-1} + r_{d-1}
1) Repeat 0 a^{d-1} times, repeat the next integer 1 a^{d-1} times and so on circularly from {0,1,...,a-1} till p_d * a^d.
2) After that, repeat 0 ceil(r_d/a) times followed by 1 ceil(r_d/a) times till we reach the nth position. Clearly, n-th integer in the sequence would not exceed a-1.
3) Every integer occurring after the position a^{d-1} p_{d-1} is increased by 1.
'''
def intervention_subroutine(A: set, k: int):
    assert type(A) == set
    assert len(A) >= 1
    assert k >= 1

    if k == 1:
        return [frozenset({v}) for v in A]
    else:
        # Setup parameters. Note that [SKDV15] use n and x+1 instead of h and L
        h = len(A)
        k_prime = min(k, h/2)
        a = math.ceil(h/k_prime)
        assert a >= 2
        L = math.ceil(math.log(h,a))
        assert pow(a,L-1) < h and h <= pow(a,L)

        # Execute labelling scheme
        S = defaultdict(set)
        for d in range(1, L+1):
            a_d = pow(a,d)
            r_d = h % a_d
            p_d = h // a_d
            a_dminus1 = pow(a,d-1)
            r_dminus1 = h % a_dminus1 # Unused
            p_dminus1 = h // a_dminus1
            assert h == p_d * a_d + r_d
            assert h == p_dminus1 * a_dminus1 + r_dminus1
            for i in range(1, h+1):
                node = A[i-1]
                if i <= p_d * a_d:
                    val = (i % a_d) // a_dminus1
                else:
                    val = (i - p_d * a_d) // math.ceil(r_d / a)
                if i > a_dminus1 * p_dminus1:
                    val += 1
                S[(d,val)].add(node)
        return [frozenset(x) for x in S.values()]

'''
Balanced partitioning on trees
'''
def tree_balanced_partitioning(nx_G, L: int):
    assert nx.is_tree(nx_G)
    assert L >= 1

    adj_list = get_adj_list(nx_G)
    V = set(nx_G.nodes())
    n = len(V)
    A = set()

    while len(V) > np.ceil(n/(L+1)):
        # Consider the subgraph on remaining vertices
        H = nx_G.subgraph(V)
        assert nx.is_tree(H)

        # Root arbitrarily
        root = list(V)[0]

        # Perform DFS
        dfs_preorder = list(nx.dfs_preorder_nodes(H, source=root))
        dfs_index = dict()
        for v in dfs_preorder:
            dfs_index[v] = len(dfs_index)
        children = defaultdict(set)
        for v in V:
            for w in adj_list[v]:
                if w in V and dfs_index[w] > dfs_index[v]:
                    children[v].add(w)

        # Compute sizes of subtrees T_u for each node u
        # Traverse in reverse DFS ordering
        subtree_size = dict()
        for v in dfs_preorder[::-1]:
            subtree_size[v] = 1
            for w in children[v]:
                subtree_size[v] += subtree_size[w]

        # if-else cases
        u = None
        for v in V:
            # Find a subtree T_u such that |T_u| = 1 + ceil(n/(L+1))
            if subtree_size[v] == 1 + np.ceil(n/(L+1)):
                u = v
                break
        if u is None:
            # Find a subtree T_u such that |T_u| > 1 + ceil(n/(L+1)) and all children w have |T_w| <= ceil(n/(L+1))
            # Start from root and recurse (see proof)
            done = False
            v = root
            while not done:
                # Check if all children w have |T_w| <= ceil(n/(L+1))
                ok = True
                for w in children[v]:
                    if subtree_size[w] > np.ceil(n/(L+1)):
                        v = w
                        ok = False
                        break
                if ok:
                    u = v
                    done = True

        # Add u to A
        assert u is not None
        A.add(u)

        # Remove T_u from V
        old_V_size = len(V)
        def remove_subtree_from_V(node):
            V.remove(node)
            for w in children[node]:
                remove_subtree_from_V(w)
        remove_subtree_from_V(u)
        new_V_size = len(V)

        # Sanity check: |V| drops by subtree_size[u]
        assert new_V_size == old_V_size - subtree_size[u]

    # Sanity check: |A| <= L and subtrees in G[V\A] have size <= ceil(n/(L+1))
    assert len(A) <= L
    G_check = nx_G.subgraph(set(nx_G.nodes()).difference(A))
    for cc_nodes in nx.connected_components(G_check):
        assert len(cc_nodes) <= np.ceil(n/(L+1))

    return A

'''
Modified from [CSB22]'s separator_policy
'''
def r_adaptive_policy(dag: DAG, r: int, k: int, verbose: bool = False) -> set:
    assert r >= 1
    assert k >= 1

    intervened_nodes = set()
    current_cpdag = dag.cpdag()
    if verbose: print(f"Remaining edges: {current_cpdag.num_edges}")

    # If we are given r > log_2(n), we will use the additional adaptivity to check whether we should skip interventions
    # The current implementation of [CSB22] is actually n-adaptive since it ALWAYS performs such checks before intervening
    n = dag.nnodes
    L = np.ceil(np.power(n, 1/r))
    checking_budget = max(0, r - int(np.ceil(np.log2(n))))
    r -= checking_budget

    # Subroutine to extract essential graph and then remove oriented arcs
    def get_essential_graph_without_oriented_arcs():
        nonlocal current_cpdag

        G = nx.Graph()
        G.add_nodes_from(current_cpdag.nodes)
        G.add_edges_from(current_cpdag.edges)
        return G

    # Subroutine to perform interventions
    def perform_one_round_of_interventions(intervention_set): 
        nonlocal intervened_nodes
        nonlocal current_cpdag
        nonlocal checking_budget

        for intervention in intervention_set:
            assert len(intervention) <= k
            relevant_vertices = [v for v in intervention]
            if checking_budget > 0:
                G = get_essential_graph_without_oriented_arcs()
                relevant_vertices = [v for v in intervention if G.degree[v] > 0]
            checking_budget -= 1
            if len(relevant_vertices) > 0:
                intervened_nodes.add(frozenset(relevant_vertices))
                current_cpdag = current_cpdag.interventional_cpdag(dag, intervention)

    # Loop for (up to) r-1 rounds
    while r > 1:
        r -= 1
        if current_cpdag.num_arcs != dag.num_arcs:
            G = get_essential_graph_without_oriented_arcs()
            intervention_set = set()
            for cc_nodes in nx.connected_components(G):
                if len(cc_nodes) > 1:
                    H = G.subgraph(cc_nodes).copy()

                    if H.size() == len(cc_nodes) * (len(cc_nodes) - 1) / 2:
                        # If H is a clique, add all vertices to intervention set
                        intervention_set.update(cc_nodes)
                    else:
                        # Compute clique tree T_H
                        T_H = compute_clique_tree(H)

                        # Compute L-balanced partitioning on T_H
                        for mc_node in tree_balanced_partitioning(T_H, L):
                            # Add all vertices involved to the maximal clique
                            intervention_set.update(T_H.nodes()[mc_node]["mc"])

            # Non-adaptive alternative (treat as final round, even though we may have more adaptivity)
            # Compute G-separating system on remaining relevant vertices
            # For atomic interventions, this corresponds to the minimum vertex cover
            relevant_nodes = set()
            for cc_nodes in nx.connected_components(G):
                if len(cc_nodes) > 1:
                    relevant_nodes.update(cc_nodes)
            minVC = minimum_vertex_cover(G.subgraph(relevant_nodes))
            assert len(set(minVC).difference(relevant_nodes)) == 0

            # Perform a round of intervention (choose the cheaper option)
            if len(minVC) <= len(intervention_set):
                # Treating as final round and assign all remaining budget for checking
                checking_budget += r
                r = 0
                perform_one_round_of_interventions(intervention_subroutine(minVC, k))
            else:
                perform_one_round_of_interventions(intervention_subroutine(intervention_set, k))

    # If still not fully oriented, perform a single round of non-adaptive interventions
    assert r <= 1
    if current_cpdag.num_arcs != dag.num_arcs:
        assert r == 1
        G = get_essential_graph_without_oriented_arcs()

        # Compute G-separating system on remaining relevant vertices
        # For atomic interventions, this corresponds to the minimum vertex cover
        relevant_nodes = set()
        for cc_nodes in nx.connected_components(G):
            # Sanity check: Number of maximal cliques in each connected component at the final non-adaptive round is at most L = ceil(n^(1/r))
            T = compute_clique_tree(G.subgraph(cc_nodes))
            assert len(T.nodes()) <= L
            if len(cc_nodes) > 1:
                relevant_nodes.update(cc_nodes)
        minVC = minimum_vertex_cover(G.subgraph(relevant_nodes))
        assert len(set(minVC).difference(relevant_nodes)) == 0

        # Perform a round of intervention
        perform_one_round_of_interventions(intervention_subroutine(minVC, k))
        
    assert current_cpdag.num_arcs == dag.num_arcs
    return intervened_nodes

