import numpy as np


Samples = np.array(
    [[1, 4, 1],
     [1, 5, 1],
     [2, 4, 1],
     [2, 5, 1], 
     [3, 1, 1],
     [3, 2, 1],
     [4, 1, 1], 
     [4, 2, 1]]
)


Targets = np.array(
    [[0],
     [0],
     [0],
     [0],
     [1],
     [1],
     [1],
     [1]]
)


test = np.array(
    [[1, 6, 1],
     [2, 6, 1],
     [5, 3, 1],
     [5, 2, 1]]
)

exp = np.array(
    [[0],
     [0],
     [1],
     [1]]
)


#learning rate

alp=[[1],
     [0.5],
     [0.25]]

theta = 0

Weights= np.zeros( Samples.shape[1] )

# Step Function
def f(val):
    if val >= theta:
        return 1
    return 0




def epoch(alpha):
    n, m = Samples.shape
    
    for i in range(n):
        global Weights,flag
        y = np.dot(Samples[i], Weights) 
        y = f(y)
        dw=np.zeros(Samples.shape[1])
        if Targets[i]!=y:
            dw=alpha*(Targets[i]-y)*Samples[i]
            flag=1
        
        Weights=Weights+dw
        
    

flag=0
n_epochs = 100 # maximum 100
for j in range(3):
    for n in range(n_epochs):
        flag=0
        epoch(alp[j])
        if flag==0:                     #if no weights were changed in this epoch then break because converged
            score=0
            print("\nHere is your training output for alpha = ",alp[j][0])
            for i in range (8):
                print("Input  :          ",Samples[i][0],Samples[i][1])
                print("Output :          ",f(np.dot(Weights,Samples[i])))
                print("Expected Output : ",Targets[i][0])
                if Targets[i][0] == f(np.dot(Weights,Samples[i])):
                    score=score+1
            print("Sample Accuracy = ",(100*score)/8,"%")
            #Here we test the trained neuron
            score=0
            print("\nHere is your testing result for alpha = ",alp[j][0])
            for i in range (4):
                print("Input  :          ",test[i][0],test[i][1])
                print("Output :          ",f(np.dot(Weights,test[i])))
                print("expected Output : ",exp[i][0])
                if exp[i][0] == f(np.dot(Weights,test[i])):
                    score=score+1
            print("Testing  Accuracy = ",(100*score)/4,"%")
            break







