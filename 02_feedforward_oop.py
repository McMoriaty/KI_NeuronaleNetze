import numpy as np

y = np.array([0,0])

def sigmaoid(x):
    x = 1/(1+(1/np.power(np.e, x)))
    return(x)

with open('C:\Schule\TalentaIt\Phyton\KI_NeuronaleNetze\data\data_dark_bright_test_4000.csv','r')as f:
    data_list = f.readlines()

class Network:  
    def __init__(self, neurons, hidden_layer):
        self.wA= np.random.uniform(-0.5,0.5,(int(neurons[0]),(int(neurons[1]))))
        self.wB= np.random.uniform(-0.5,0.5,(int(hidden_layer[0]),(int(hidden_layer[1]))))
        
    def feedforward(self, k):
        h=sigmaoid(np.dot(self.wA, k)) 
        y=sigmaoid(np.dot(self.wB, h))
        return(y)

    def run(self,data_list):
        self.correct=0
        self.runtotal=0
        for i in range(len(data_list)):
            t = data_list[i].split(",")
            TargetOutput = int(t[0])
            wert_test = np.array([int(t[1])/255, int(t[2])/255, int(t[3])/255, int(t[4])/255])
            wert_test = wert_test.reshape((-1,1))
            y=self.feedforward(wert_test)
            self.check(y,TargetOutput)     

        print(self.average)

    def check(self,y,TargetOutput):
        n,o=y
        if n<o:
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

        self.average=100/self.runtotal*self.correct
        
network=Network([3,4],[2,3])
network.run(data_list)
    