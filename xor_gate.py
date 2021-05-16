import numpy as np
import math

sample = np.array(
    [[0, 0, 1],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 1]])  #  add one extra for bias

target = np.array(
    [[0],
     [1],
     [1],
     [0]])

eta = 1  # Learning rate

V = np.array([[0, 0, 0], [0, 0, 0]])  # Weight matrix from input to hidden; may edit this
W = np.array([0, 0, 0])               # Weight matrix from hidden to output; may edit this
# print(V, W)


n_epochs = 2
print("\nHere is your XOR gate\n")
for j in range(4): 
    # print("epoch number : ",k+1)
    for k in range(n_epochs):
        yin=np.zeros(2)


        yin[0] = np.dot(sample[j], V[0])
        yin[1] = np.dot(sample[j], V[1])

        g =np.vectorize(lambda yin: yin)
        #g = np.vectorize(lambda yin: 1 / (1 + math.exp(-yin)))  # Activation function; may be different
        #h = np.vectorize(lambda yin: g(yin) * (1 - g(yin)))  # Derivative of g(sample)
        h=np.vectorize(lambda yin: 1)

        inp=np.zeros(3)
        inp[0]=g(yin[0])
        inp[1]=g(yin[1])
        inp[2]=1

        yout=np.dot(inp,W)
        youtFinal=g(yout)

        #yout= 1 / (1 + math.exp(-yin))
        # print("input",sample[j])
        # print("output : " , youtFinal)

        err=(1)*(target[j]-youtFinal)#(target[j]-youtFinal)

        deltaOut=h(yout)*(err)

        dW = eta*deltaOut*inp

        # print("\ninner weights",V) # updated hidden weight
        # print("\nWeights outer",W) # updated outer weight

        W=W+dW


        deltaIn=np.zeros(2)
        deltaIn[0]=deltaOut*W[0]*h(yin[0])
        deltaIn[1]=deltaOut*W[1]*h(yin[1])



        V[0][0]= V[0][0] + eta*deltaIn[0]*sample[j][0]
        V[0][1]= V[0][1] + eta*deltaIn[1]*sample[j][0]
        V[1][0]= V[1][0] + eta*deltaIn[0]*sample[j][1]
        V[1][1]= V[1][1] + eta*deltaIn[1]*sample[j][1]
        V[0][2]= V[0][2] + eta*deltaIn[0]
        V[1][2]= V[1][2] + eta*deltaIn[1]

        #print("hidden",V)
        #print("outer",W)
        
        if k + 1 == n_epochs:
            print("{0}\t{1}\t{2} ".format(sample[j][0],sample[j][1],youtFinal))
            
