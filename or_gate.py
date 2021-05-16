import numpy as np

Samples = np.array(
    [[0, 0, 1],
     [0, 1, 1],
     [1, 0, 1], 
     [1, 1, 1]]
)

Targets = np.array(
    [[0],
     [1],
     [1],
     [1]]
)

#learning rate
alp = 1

theta = 0

Weights = np.zeros( Samples.shape[1] )


# Step Function
def f(val):
    if val >= theta:
        return 1
    return 0

def epoch():
    n, m = Samples.shape
    
    for i in range(n):
        global Weights,flag
        y = np.dot(Samples[i], Weights) 
        y = f(y)
        dw=np.zeros(Samples.shape[1])
        if Targets[i]!=y:
            dw=alp*(Targets[i]-y)*Samples[i]
            flag=1
        
        Weights=Weights+dw
        
    

flag=0
n_epochs = 100 # maximum 100
for n in range(n_epochs):
    flag=0
    epoch()
    print("Weights after epoch : ",n+1)
    print(Weights)
    if flag==0:                     #if no weights were changed in this epoch then break because converged
        break

print("\nHere is your OR gate\n")
for i in range (4):
    print("{0}\t{1}\t{2} ".format(Samples[i][0],Samples[i][1],f(np.dot(Weights,Samples[i]))))

 

