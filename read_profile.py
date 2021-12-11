import pstats
from pstats import SortKey
p = pstats.Stats('profiling_stats')
# p.strip_dirs().sort_stats(True).print_stats(20)
p.sort_stats(True).print_stats(20)