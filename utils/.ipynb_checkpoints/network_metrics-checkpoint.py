import torch
import sys
import networkx as nx
from networkx.algorithms import approximation, mis
import matplotlib.pyplot as plt
import re

from max_set import get_max_degree_concurrency

INFERENCE_SIZE = (224, 224)

def remove_accumulate_nodes(graph):
    new_graph = graph.copy()

    for node in graph.nodes():
        nodename = re.sub(r'[0-9]+', '', str(node))
        print(nodename)
        if nodename.endswith('AccumulateGrad'):
            if len(nx.algorithms.dag.ancestors(graph, node)) == 0 or \
                len(nx.algorithms.dag.descendants(graph, node)) == 0:
                new_graph.remove_node(node)
        elif nodename.endswith('TBackward'):
            if len(nx.algorithms.dag.ancestors(graph, node)) == 0 or \
                len(nx.algorithms.dag.descendants(graph, node)) == 0:
                new_graph.remove_node(node)
        elif nodename.endswith('CheckpointFunctionBackward'):
            if len(nx.algorithms.dag.ancestors(graph, node)) == 0 or \
                len(nx.algorithms.dag.descendants(graph, node)) == 0:
                new_graph.remove_node(node)
        elif nodename.endswith('Tensor'):
            if len(nx.algorithms.dag.ancestors(graph, node)) == 0 or \
                len(nx.algorithms.dag.descendants(graph, node)) == 0:
                new_graph.remove_node(node)

    return new_graph

def get_model_graph(model):

    OLD_RECURSION_LIMIT = sys.getrecursionlimit()
    sys.setrecursionlimit(1500)
    dummy_input = torch.randn(1, 3, INFERENCE_SIZE[0], INFERENCE_SIZE[1])
    dummy_output = model(dummy_input)
    params = model.state_dict()
    param_map = {id(v): k for k, v in params.items()}

    seen = set()

    nodes = {}

    edges = {}

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def _getname(tensor):
        name = type(tensor).__name__
        return f"{str(id(tensor))}-{name}"

    def add_edge(f, t):
        fromid = _getname(f)
        if fromid in edges:
            edges[fromid].append(_getname(t))
        else:
            edges[fromid] = [_getname(t)]

    def add_nodes(var, d=0):
        if var not in seen:
            if torch.is_tensor(var):
                nodes[_getname(var)] = size_to_str(var.size())
            elif hasattr(var, 'variable'):
                subvar = var.variable
                node_name = '%s\n %s' % (param_map.get(id(subvar)), size_to_str(subvar.size()))
                nodes[_getname(var)] = node_name
            else:
                nodes[_getname(var)] = str(type(var).__name__)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_edge(u[0], var)
                        if u[0] not in seen:
                            add_nodes(u[0], d+1)
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_edge(t, var)
                    if t not in seen:
                        add_nodes(t, d+1)

    try:
        add_nodes(dummy_output.grad_fn)
    except BaseException:
        print("Exception generated on graph")
        # some models have a weird structure on the output
        add_nodes(dict(dummy_output)['out'].grad_fn)

    # print("Model graph complete", len(edges), len(nodes), flush=True)
    sys.setrecursionlimit(OLD_RECURSION_LIMIT)
    return nodes, edges

def __topo_sort(edge_lookup, nodeid, visited, stack):
    visited[nodeid] = True
    node_edges = edge_lookup[nodeid] if nodeid in edge_lookup else []
    for node in node_edges:
        if node not in visited:
            __topo_sort(edge_lookup, node, visited, stack)
    stack.insert(0, nodeid)

def topological_sort(e):
    # print("Sort edge map", flush=True)
    visited = {}
    stack = []
    for idx, nodeid in enumerate(e.keys()):
        if nodeid not in visited:
            __topo_sort(e, nodeid, visited, stack)
    # print("Edge map sorted", flush=True)
    return stack

def get_critical_path(model):
    nodes, edges = get_model_graph(model)
    from_nodes_sorted = topological_sort(edges)
    latencies = {}

    for node in from_nodes_sorted:
        e = edges[node] if node in edges  else []
        if node not in latencies:
            latencies[node] = 0
        for edge in e:
            edge_latency = latencies[edge] if edge in latencies else 0
            latencies[edge] = max(edge_latency, latencies[node] + 1)
    return max(latencies.values()), latencies, [latencies[k] for k in from_nodes_sorted]


def pytorch_to_nx(model):
    G = nx.DiGraph()
    nodes, edges = get_model_graph(model)

    G.add_nodes_from(nodes)

    for parent, children in edges.items():
        for child in children:
            G.add_edge(parent, child)

    G = remove_accumulate_nodes(G)
    return G

def maximal_set_on_path(model):
  print("Getting max set on path")
  graph = pytorch_to_nx(model)
  # first add root node if none exists
  graph_copy = graph.copy()

  topo_sort = nx.topological_sort(graph_copy)

  print("Got topological sort", flush=True)

  path_lengths = {} # node -> distance
  current_cdl = 0

  for node in topo_sort:
    if node not in path_lengths.keys():
      path_lengths[node] = 0

    for parent in graph_copy.predecessors(node):
      if parent not in path_lengths.keys():
        path_lengths[parent] = 0
      # node is one away from the ancestor
      output_length = max(path_lengths[node], path_lengths[parent] + 1)
      path_lengths[node] = output_length

  print("Parsed decendents and ancestors", flush=True)

  # path_length -> node_set
  independent_sets = {}
  for node, length in path_lengths.items():
    if length not in independent_sets.keys():
      independent_sets[length] = set()

    independent_sets[length].add(node)

  print("Got all independent sets at each CDL", flush=True)

  # trivial to initialize tensors that are all at the zeroth position
  del independent_sets[0]

  return max([len(X) for X in independent_sets.values()])

def longest_path(model):
    print("Computing maximal antichain, may be slow")
    graph = pytorch_to_nx(model)
    return nx.algorithms.dag.dag_longest_path_length(graph)

def maximal_antichain(model):
    print("Computing maximal antichain, may be slow")
    graph = pytorch_to_nx(model)
    print("Graph conversion complete", len(graph.nodes()), len(graph.edges()), flush=True)
    all_antichains = nx.antichains(graph)
    print("got all antichains", flush=True)
    chain_lengths = map(lambda antichain: (len(antichain), antichain), all_antichains)
    print("Computed chain lengths", flush=True)
    # print(list(set(map(lambda item: item[0], chain_lengths))))
    max_len = 0
    idx = 0
    for length, _ in chain_lengths:
        idx += 1
        max_len = max(length, max_len)
        print(f"{idx}", flush=True, end='\r')
    # max_len = max(map(lambda item: item[0], chain_lengths))
    print("Got maximum length =", max_len, flush=True)
    longest_chains = map(lambda item: set(item[1]), list(filter(lambda item: item[0] == max_len, chain_lengths)))

    return max_len, list(longest_chains)

def approx_max_ind_set(model):
    print("Computing approximate independent set")
    graph = pytorch_to_nx(model)
    # nx.draw_planar(graph, font_size=8, with_labels=True)
    # plt.show()
    print("Graph conversion complete", len(graph.nodes()), len(graph.edges()), flush=True)
    mdc, _ = get_max_degree_concurrency(graph)
    print(mdc)
    print('got max', mdc)
    return mdc, []

def total_nodes(model):
  graph = pytorch_to_nx(model)
  # first add root node if none exists
  graph_copy = graph.copy()

  return len(graph_copy.nodes())

def total_edges(model):
  graph = pytorch_to_nx(model)
  # first add root node if none exists
  graph_copy = graph.copy()

  return len(graph_copy.edges())
