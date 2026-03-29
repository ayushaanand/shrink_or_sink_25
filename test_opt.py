import operator
N = 96
FLIP_H_IDX = [i for i in range(N)] + [i + N for i in range(N)]
get_h = operator.itemgetter(*FLIP_H_IDX)
state = tuple(range(192))
t1 = tuple(state[i] for i in FLIP_H_IDX)
t2 = get_h(state)
print(t1 == t2)
