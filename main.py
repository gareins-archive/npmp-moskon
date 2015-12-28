import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

if __name__ == "__main__":
    def eq(X, t):
        x, y = X
        return [x - y - np.exp(t), x + y + 2 * np.exp(t)]

    init = [-1.0, -1.0]
    t = np.linspace(0, 4, num=50)
    X = odeint(eq, init, t)

    x = X[:, 0]
    y = X[:, 1]

    plt.plot(t, x, 'k--')
    plt.plot(t, y, 'k:')

    plt.show()
