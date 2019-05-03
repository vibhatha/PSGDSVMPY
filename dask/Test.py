import numpy as np
import dask.array as da
x = np.random.rand(50000,50000)
y = da.from_array(x, chunks=(100))
ans = y.mean().compute()
print(ans)
