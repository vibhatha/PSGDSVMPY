# prepare python
import numpy as np
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure


# Create four populations of 100 observations each
pop1_X, pop1_Y = make_regression(n_samples=100, noise=20, n_informative=1, n_features=1, random_state=1, bias=0)
pop2_X, pop2_Y = make_regression(n_samples=100, noise=20, n_informative=1, n_features=1, random_state=1, bias=100)
pop3_X, pop3_Y = make_regression(n_samples=100, noise=20, n_informative=1, n_features=1, random_state=1, bias=-100)

# Stack them together
pop_X = np.concatenate((pop1_X, pop2_X, pop3_X))
pop_Y = np.concatenate((pop1_Y, 2 * pop2_Y, -2 * pop3_Y))

# Add intercept to X
pop_X = np.append(np.vstack(np.ones(len(pop_X))), pop_X, 1)

# convert Y's into proper column vectors
pop_Y = np.vstack(pop_Y)

### plot
mycmap = cm.brg
fig = plt.figure(figsize=(6, 6), dpi=150)
plt.subplots_adjust(hspace=.5)
gridsize = (1, 1)
ax0 = plt.subplot2grid(gridsize, (0, 0))
sc = ax0.scatter(pop_X[:, 1], pop_Y, s=100, alpha=.4, c=range(len(pop_X)), cmap=mycmap)
plt.colorbar(sc, ax=ax0)
plt.show()
