import numpy as np





with open('data','data_dark_bright_test_4000.csv','r')as f:
    int(data_list) = f.readlines()

input_layer=([[44],[14],[27]])

class Network:  
    def __init__(self):
        self.wA=np.array([[-0.3,-0.7,-0.9,-0.9],[-1,-0.6,-0.6,-0.6],[0.80,0.50,0.70,0.8]])
        self.wB=np.array([[2.6,2.1,-1.2],[-2.3,-2.3,1.1]])
        

        def sigmaoid(self):
            self.sigmaoid=1/(1+np.e**-self.x)
        
        def feedforward(self):
            h=sigmaoid(self.wA*input_layer) ##hiddenlayer
            y=sigmaoid(self.wB*h) ##ourputlayer

            




