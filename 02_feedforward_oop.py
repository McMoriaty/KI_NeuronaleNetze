import numpy as np

y = np.array([0,0])

def sigmaoid(x):
    x = 1/(1+(1/np.power(np.e, x)))
    return(x)

with open('C:\Schule\TalentaIt\Phyton\KI_NeuronaleNetze\data\data_dark_bright_test_4000.csv','r')as f:
    data_list = f.readlines()

class Network:  
    def __init__(self, wA, wB):
        self.wA= wA 
        self.wB= wB 
        
    def feedforward(self, k):
        h=sigmaoid(np.dot(self.wA, k)) 
        self.y=sigmaoid(np.dot(self.wB, h))

    def check(self):
        if network.y[0]<network.y[1]:
            b=1
            if b == TargetOutput:
                self.correct += 1
                self.runtotal += 1
            else:
                self.runtotal += 1
        else:
            b=0
            if b == TargetOutput:
                self.correct += 1
                self.runtotal += 1
            else:
                self.runtotal += 1

network=Network(np.random.rand(3,4),np.random.rand(2,3))
network.correct=0
network.runtotal=0
for i in range(len(data_list)):
    t = data_list[i].split(",")
    TargetOutput = int(t[0])
    wert_test = np.array([int(t[1])/255, int(t[2])/255, int(t[3])/255, int(t[4])/255])
    wert_test = wert_test.reshape((-1,1))
    network.feedforward(wert_test)
    network.check()

print(100/network.runtotal*network.correct)
    
    


