import timeit

cy = timeit.timeit('''sgd.test(200000)''',setup='import sgd',number=10)
py = timeit.timeit('''sgdn.test(200000)''',setup='import sgdn', number=10)

print(cy, py)
print('Cython is {}x faster'.format(py/cy))
