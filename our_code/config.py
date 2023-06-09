import os
import seaborn as sns

BASE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_FOLDER, 'data')
FIGURE_FOLDER = os.path.join(BASE_FOLDER, 'figures')

policies = [
    'nu',
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
POLICY2COLOR = dict(zip(policies, sns.color_palette('bright')))
POLICY2LABEL = {
    'nu': 'Verification number',
    'adaptive_r1': 'r = 1 (non-adaptive)',
    'adaptive_r2': 'r = 2',
    'adaptive_r3': 'r = 3',
    'adaptive_rlogn': 'r = log n',
    'adaptive_r2logn': 'r = 2 log n (has checks)',
    'adaptive_r3logn': 'r = 3 log n (has checks)',
    'adaptive_rn': 'r = n (has checks)',
    'separator': 'CSB22 (has checks)',
    'separator_no_check': 'CSB22'
}
