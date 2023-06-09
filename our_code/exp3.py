from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = [
    'adaptive_r1',
    'adaptive_r2',
    'adaptive_r3',
    'adaptive_rlogn',
    'adaptive_r2logn',
    'adaptive_r3logn',
    'adaptive_rn',
    'separator',
    'separator_no_check'
]
nnodes_list = list(range(10, 101, 5))
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.GNP_TREE,
    dict(p=0.03, figname="exp3"),
    algorithms=algs,
    overwrite=False
)


