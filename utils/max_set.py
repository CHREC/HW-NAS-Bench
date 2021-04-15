import networkx as nx
import matplotlib.pyplot as plt

def get_max_degree_concurrency(graph, verbose=True, debug=False, draw=True):

  if debug or draw:
    nx.draw_spring(graph, font_size=16, with_labels=True)
    plt.show()

  print("Get all")
  topo_sorts = nx.all_topological_sorts(graph)
  print("Done")

  MDC = 0

  contiguous_sets = {}

  iter = 0
  for topo_sorted in topo_sorts:
    iter1 = 0
    iter += 1
    # print(f"{iter}", end='')
    root = topo_sorted[0]

    last_node = root

    if debug:
      print(topo_sorted)

    independent_sets = []
    parallel_tasks = set()

    concurrency = 0

    for idx, node in enumerate(topo_sorted): # iterate through all possible topological sorts (NP)
      iter1 += 1
      print(f"{iter}.{iter1}", flush=True, end='\r')
      # parallel_tasks.add(node) # add new node to set of parallel tasks

      if idx > 1:
        previous_node = topo_sorted[idx - 1] # check previous topological ordering
        inter_dependent = False

        if not nx.has_path(graph, source=previous_node, target=node): # dependency not detected
          parallel_tasks.add(previous_node)
          # check other parallel tasks for dependencies
          for task in parallel_tasks:
            if nx.has_path(graph, source=task, target=node): # cant add this one
              if debug:
                print(f"Dependent on other task {node} -> {parallel_tasks}")
              inter_dependent = True
              break
        else:
          if debug:
            print(f"Dependent on previous node {node} -> {previous_node}")
          inter_dependent = True

        if not inter_dependent:
          parallel_tasks.add(node) # task can be executed in parallel
          if debug:
            print(f"Parallel {previous_node} || {node}")
            print(parallel_tasks)
        else:
          if debug:
            print(f"Dependency {previous_node} -> {node}")
            print(parallel_tasks)
          if len(parallel_tasks) > 0:
            independent_sets.append(parallel_tasks.copy())
          parallel_tasks.clear()
          parallel_tasks.add(node)

    if debug:
      print(independent_sets)

    for task_group in independent_sets:
      concurrency = max(concurrency, len(task_group))

    max_tasks = []

    for group in independent_sets:
      if len(group) == concurrency:
        max_tasks.append(group)

    if debug:
      print(f"backcheck done ------------ {concurrency}")
      print(max_tasks)
    if concurrency > 0:
      if concurrency not in contiguous_sets.keys():
        contiguous_sets[concurrency] = [max_tasks]
      else:
        is_unique = True
        for tasks in contiguous_sets[concurrency]:
          if is_unique:
            for existing_group in tasks:
              if is_unique:
                for parallel_group in max_tasks:
                  if len(existing_group.difference(parallel_group)) == 0: # sets are disjoint, different set of tasks
                    is_unique = False
                    break
          else:
            break
        if is_unique:
          contiguous_sets[concurrency] += [max_tasks]
          if debug:
            print("unique task set found", max_tasks)

  if debug:
    print(contiguous_sets)

  MDC = max(contiguous_sets.keys())

  if verbose:
    print(f"MDC = {MDC}, {contiguous_sets[MDC]}")

  return MDC, contiguous_sets[MDC]