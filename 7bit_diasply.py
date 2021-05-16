import numpy as np

Samples = np.array(
    [[1, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 0, 0, 0, 0],
     [1, 1, 0, 1, 1, 0, 1],
     [1, 1, 1, 1, 0, 0, 1],
     [0, 1, 1, 0, 0, 1, 1],
     [1, 0, 1, 1, 0, 1, 1],
     [1, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 0, 1, 1]]
)

Targets = np.diag(np.ones(Samples.shape[0]))

eta = 1
theta = 6
Weights = np.zeros((Samples.shape[1], Targets.shape[1]))


#activation function
def f(val):
    if val > theta:
        return 1
    return 0


activation = np.vectorize(f)


def epoch():
    n, m = Samples.shape
    fl = True
    for i in range(n):
        y = activation(Samples[i] @ Weights)
        for j in range(Weights.shape[0]):
            dw = eta * Samples[i][j] * (Targets[i] - y)
            if np.any(dw != 0):
                fl = False
            Weights[j] += dw
    return fl


n_epochs = 0
flag = False
print("Initial Weights: \n", Weights)
while not flag:
    flag = epoch()
    n_epochs += 1
    # print(Weights)

print("\nFinal Weights: \n", Weights)
print("\nRow wise result of each sample, all element in row represent final value of output of each neuron :\n",activation(Samples @ Weights))
print("\nConvergence achieved in", n_epochs, "epochs") 