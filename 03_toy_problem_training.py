import numpy as np

y = np.array([0,0])

def sigmaoid(x):
    x = 1/(1+(1/np.power(np.e, x)))
    return(x)

with open('C:\Schule\TalentaIt\Phyton\KI_NeuronaleNetze\data\data_dark_bright_test_4000.csv','r')as f:
    data_test_list = f.readlines()

with open('C:\Schule\TalentaIt\Phyton\KI_NeuronaleNetze\data\data_dark_bright_training_20000.csv','r')as f:
    data_training_list = f.readlines()


class Network:  
    def __init__(self, neurons, hidden_layer,learning_rate):
        self.wA= np.random.uniform(-0.5,0.5,(int(neurons[0]),(int(neurons[1]))))
        self.wB= np.random.uniform(-0.5,0.5,(int(hidden_layer[0]),(int(hidden_layer[1]))))
        self.learning_rate=learning_rate
        
        
    def feedforward(self, k):
        self.h=sigmaoid(np.dot(self.wA, k)) 
        y=sigmaoid(np.dot(self.wB, self.h))
        return(y)

    def costfunction(self,Eout):
        c= 0.5*(Eout[0]**2+Eout[1]**2)
        return(c)

    def training(self,data):
        self.correct=0
        self.runtotal=0
        for i in range(len(data)):
            t = data[i].split(",")
            TargetOutput = int(t[0])
            wert_test = np.array([int(t[1])/255, int(t[2])/255, int(t[3])/255, int(t[4])/255])
            wert_test = wert_test.reshape((-1,1))
            y=self.feedforward(wert_test)
            self.check(y,TargetOutput)
            c = self.costfunction(y)
            
            if c >=0:
                E_out=TargetOutput-y
                E_hidden=np.dot(self.wB.T,E_out)

                #New wB value
                self.wBnew = self.wB+self.learning_rate*(np.dot((E_out*y*(1-y)), self.h.T))
                self.wB=self.wBnew

                #New wA value
                self.wAnew= self.wA+self.learning_rate*(np.dot((E_hidden*self.h*(1-self.h)), wert_test.T))
                self.wA=self.wAnew
        
        print(self.wA)   
        print(self.wB)   

    def test(self,data):
        self.correct=0
        self.runtotal=0
        for i in range(len(data)):
            t = data[i].split(",")
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
        
network=Network([3,4],[2,3],0.1)
network.training(data_training_list)
network.test(data_test_list)
    