import itertools
lst = [1, 2, 3]
#lst = [1, 2, 3, 4, 5, 6, 7, 8]
for i in xrange(1, len(lst)+1):
    for j in [list(x) for x in itertools.combinations(lst, i)]:
		print(' '.join(str(e) for e in j))
