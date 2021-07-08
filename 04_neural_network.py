import numpy as np

y = np.array([0,0])

def sigmaoid(x):
    q = 1/(1+(1/np.power(np.e, x)))
    return(q)

with open('C:\Schule\TalentaIt\Phyton\KI_NeuronaleNetze\data\mnist_test.csv','r')as f:
    data_test_list = f.readlines()

with open('C:\Schule\TalentaIt\Phyton\KI_NeuronaleNetze\data\mnist_train.csv','r')as f:
    data_training_list = f.readlines()

class Network:  
    def __init__(self, neurons, hidden_layer,learning_rate):
        self.wA= np.random.uniform(-0.5,0.5,(int(neurons[0]),(int(neurons[1]))))
        self.wB= np.random.uniform(-0.5,0.5,(int(hidden_layer[0]),(int(hidden_layer[1]))))
        self.learning_rate=learning_rate
        
    def feedforward(self,k):
        h = sigmaoid(np.dot(self.wA, k)) 
        y = sigmaoid(np.dot(self.wB, h))
        return h,y

    def training(self,data):
        self.correct=0
        self.runtotal=0
        for i in range(len(data)):
            t = data[i].split(",")
            TargetOutput = int(t[0])
            wert_test = np.asfarray(t[1:len(t)])/255
            wert_test = wert_test.reshape((-1,1))
            h,y=self.feedforward(wert_test)
            self.check(y,TargetOutput)

            Target=[]
            for i in range(10):
                if i==TargetOutput:
                    Target.append(1)
                else:
                    Target.append(0)

            Target=np.array(Target)
            Target = Target.reshape((-1,1))

            E_out=Target-y
            E_hidden=np.dot(self.wB.T,E_out)

            #New wB value
            self.wB += self.learning_rate*(np.dot((E_out*y*(1-y)), h.T))

            #New wA value
            self.wA += self.learning_rate*(np.dot((E_hidden*h*(1-h)), wert_test.T))
      
        print(self.wA)   
        print(self.wB)   

    def test(self,data):
        self.correct=0
        self.runtotal=0
        for i in range(len(data)):
            TargetOutput=0
            t = data[i].split(",")
            TargetOutput = int(t[0])
            wert_test = np.asfarray(t[1:len(t)])/255
            wert_test = wert_test.reshape((-1,1))
            h,y=self.feedforward(wert_test)
            self.check(y,TargetOutput)

        print(self.correct*100/self.runtotal)

    def check(self,y,TargetOutput):
        hy = np.array(y).tolist()
        highest_number = max(hy)
        number_right = hy.index(highest_number)
        if number_right== TargetOutput:
            self.correct += 1
            self.runtotal += 1
        else:
            self.runtotal += 1
        
network=Network([200,784],[10,200],0.035)
network.training(data_training_list)
network.test(data_test_list)