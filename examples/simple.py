import networkx as nx
import rustpyppr
# import random
from pprint import pprint
from itertools import chain
import time

# def create_random_graph(size):
#     d = {i: [] for i in range(size)}
#     for x in range(size):
#         for y in random.sample(range(x+1, size), k=min(60, size-x-1)):
#             d[x].append(y)
#             d[y].append(x)
#
#     return d


def time_func(f, args, kwargs):
    t0 = time.perf_counter()
    res = f(*args, **kwargs)
    t1 = time.perf_counter()
    print(f'>>> function {f.__name__} from {f.__module__} took {t1-t0:.6f} seconds')
    return res


def print_compare(*results):
    nodes = set(chain.from_iterable(r.keys() for r in results))
    nodes = sorted(nodes)
    for n in nodes:
        print(f'{n:03d}\t' + '\t'.join(f'{r.get(n, 0.0):.4f}' for r in results))


def compare_values(g, edge_dict):
    nx_res = nx.pagerank(g, personalization={0: 1.0})
    rust_res1 = rustpyppr.forward_push(edge_dict, 0, 0.85, 1e-6)
    rust_res2 = rustpyppr.forward_push_vec(edge_dict, 0, 0.85, 1e-6)
    rust_res3 = rustpyppr.forward_push_vec_lazy(edge_dict, 0, 0.85, 1e-6)
    rust_res4 = rustpyppr.multiple_forward_push_vec(edge_dict, [0], 0.85, 1e-6)[0]
    rust_res5 = rustpyppr.multiple_forward_push_vec_lazy(edge_dict, [0], 0.85, 1e-6)[0]
    print_compare(nx_res, rust_res1, rust_res2, rust_res3, rust_res4, rust_res5)




def compare_time(g, edge_dict):
    rust_params = (edge_dict, 0, 0.85, 1e-3)
    rust_multi_params = (edge_dict, [0], 0.85, 1e-3)
    nx_params = {'G': g, 'personalization': {0: 1.0}}

    time_func(nx.pagerank, (), nx_params)
    time_func(rustpyppr.forward_push, rust_params, {})
    time_func(rustpyppr.forward_push_vec, rust_params, {})
    time_func(rustpyppr.forward_push_vec_lazy, rust_params, {})
    time_func(rustpyppr.multiple_forward_push_vec, rust_multi_params, {})
    time_func(rustpyppr.multiple_forward_push_vec_lazy, rust_multi_params, {})


def test_complete_small():
    size = 5
    g = nx.complete_graph(size)
    print(g)
    edge_dict = nx.to_dict_of_lists(g)
    compare_values(g, edge_dict)
    compare_time(g, edge_dict)


def test_caveman():
    g = nx.relaxed_caveman_graph(1000, 5, 0.1, seed=42)
    print(g)
    edge_dict = nx.to_dict_of_lists(g)
    compare_values(g, edge_dict)
    compare_time(g, edge_dict)


if __name__ == '__main__':
    test_complete_small()
    test_caveman()
