import timeit

cy = timeit.timeit('''example_cy.test(100,100000)''',setup='import example_cy',number=100)
py = timeit.timeit('''example.test(100,100000)''',setup='import example', number=100)

print(cy, py)
print('Cython is {}x faster'.format(py/cy))
