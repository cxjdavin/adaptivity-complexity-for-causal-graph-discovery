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
nnodes_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.HAIRBALL_PLUS_PROPORTIONAL,
    dict(degree_prop=0.4, e_min_prop=0.2, e_max_prop=0.5, figname="exp2"),
    algorithms=algs,
    overwrite=False
)


