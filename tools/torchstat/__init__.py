__copyright__ = 'Copyright (C) 2018 Swall0w'
__version__ = '0.0.7'
__author__ = 'Swall0w'
__url__ = 'https://github.com/Swall0w/torchstat'

from .compute_memory import compute_memory
from .compute_madd import compute_madd
from .compute_flops import compute_flops
from .stat_tree import StatTree, StatNode
from .model_hook import ModelHook
from .reporter import report_format
from .statistics import stat, ModelStat

__all__ = ['report_format', 'StatTree', 'StatNode', 'compute_madd',
           'compute_flops', 'ModelHook', 'stat', 'ModelStat', '__main__',
           'compute_memory']
