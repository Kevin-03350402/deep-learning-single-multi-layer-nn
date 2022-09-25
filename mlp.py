"""
Fall 2022, 10-417/617
Assignment-1

IMPORTANT:
    DO NOT change any function signatures

September 2022
"""


import numpy as np
import pickle as pk
from matplotlib import pyplot as plt 


def random_weight_init(input, output):
    b = np.sqrt(6)/np.sqrt(input+output)
    return np.random.uniform(-b, b, (output, input))

def zeros_bias_init(outd):
    return np.zeros((outd, 1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(14)] for lab in labels])


class Transform:
    """
    This is the base class. You do not need to change anything.

    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass

class ReLU(Transform):
    """
    ReLU non-linearity, combined with dropout
    IMPORTANT the Autograder assumes these function signatures
    """
    def __init__(self, dropout_probability=0):
        Transform.__init__(self)
        self.dropout_probability = dropout_probability
        

    def forward(self, x, train=True):
        # IMPORTANT the autograder assumes that you call np.random.uniform(0,1,x.shape) exactly once in this function
        """
        x shape (indim, batch_size)
        """
        # use relu
        
        relu = np.maximum(0,x)
        
        if train == True:
                # generate a mask that serve as the dropout
                self.mask = np.random.uniform(0,1,x.shape)

                # compare the mask with the threshold
                self.threshold = (self.mask>self.dropout_probability).astype(int)
                
                # element wise multiple the mask onto the input 
                dropped = np.multiply(relu,self.threshold)
                self.dropped=dropped
                return dropped
            
        if train == False:
                return (1-self.dropout_probability)*relu

        

        

        

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """

        return (np.multiply((self.dropped>0).astype(int),grad_wrt_out))

class LinearMap(Transform):
    """
    Implement this class
    For consistency, please use random_xxx_init() functions given on top for initialization
    """
    def __init__(self, indim, outdim, alpha=0, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """

        Transform.__init__(self)
        self.indim = indim
        self.outdim = outdim
        self.alpha = alpha
        self.lr = lr
        # initialize the old gradient for w and b
        self.notini = True




    def forward(self, x):

        

        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        self.indim = x.shape[0]
        
        tw = self.getW()
        tb = self.getb()
        # matrix multiplication
        res = (np.add(np.matmul(tw, x),tb))
        # store input
   
        self.input = x

        return res



    def backward(self, grad_wrt_out):
        # grad_wrt_out = dl/do
        # dl/dh = w.t * dl/do
 
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        Your backward call should accumulate gradients.
        """
        self.outdim = grad_wrt_out.shape[0]
        
        tw = self.w
        wtranspose = np.transpose(tw)
        # find dl/dh
        dldh = np.matmul(wtranspose,grad_wrt_out)


        # the gradient wrt bias is the sum of grad_wrt_out, rowwise
        
        # before we update the gradient wrt w and b, we store them in the old parameters
        # if we have never initialize the old parameters, do it:
        if self.notini == True:
            indim = self.indim
            outdim = self.outdim
            # maintain the size
            self.oldw = np.zeros((outdim,indim))
            self.oldb = np.zeros((outdim,1))
            self.notini = False
        else:
            self.oldw = np.copy(self.gw)
            self.oldb = np.copy(self.gb)
        
        # then update

   
        self.gb = np.sum(grad_wrt_out, axis=1)
        self.gb = self.gb.reshape(grad_wrt_out.shape[0],1)
  
        
   

        # transpose x
        xtrans = np.transpose(self.input)
        # find out gradient wrt w
        self.gw = np.matmul(grad_wrt_out,xtrans)
        
     
        return dldh

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        alpha = self.alpha
        oldw = self.oldw
        oldb = self.oldb
        

        # use momentum to calcuate gw and gb

        gw = np.add(alpha*oldw, self.gw)
    
        gb = np.add(alpha*oldb, self.gb)
        
        # update
        
        self.w = self.w - self.lr*gw

        self.b =  self.b - self.lr*gb

      


 

    def zerograd(self):
        pass

    def getW(self):
        """
        return W shape (outdim, indim)
        """
        # read in w
        return self.w

    def getb(self):
        """
        return b shape (outdim, 1)
        """
        
        
        return self.b


    def loadparams(self, w, b):

        self.w = w
        self.b = b
        

class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (num_classes,batch_size)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
    
        self.z = np.exp(logits)

        self.softres = self.z/np.sum(self.z,axis=0)
        self.y = labels
        

        self.batch = logits.shape[1]
        logpi = np.log(self.softres)
        
       
        loss = np.sum(-(np.multiply(labels,logpi)))
        loss = loss/self.batch
        
        return loss
        

    def backward(self):
        """
        return shape (num_classes,batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        return 1/self.batch*(self.softres - self.y)

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        totalmiss = 0
        # in each batch
        for i in range (self.softres.shape[1]):
            yhat = self.softres[:,i]
            ylabel = self.y[:,i]
            
            # find the maximum index in yhat
            index = yhat.argmax()
            
            if (ylabel[index] == False):
                totalmiss+=1
        return totalmiss
            
            


class SingleLayerMLP(Transform):
    """
    Implement this class
    """
    def __init__(self, indim, outdim, hiddenlayer=100, alpha=0.1, dropout_probability=0, lr=0.01):
        Transform.__init__(self)
        self.linear1 = LinearMap(indim, outdim, alpha, lr)
        self.activation = ReLU(dropout_probability)
        self.linear2 = LinearMap(indim, outdim, alpha, lr)
        

    def forward(self, x, train=True):
        """
        x shape (indim, batch_size)
        """
        x = self.linear1.forward(x)
        x = self.activation.forward(x,train)
        x = self.linear2.forward(x)

        return x

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        grad = self.linear2.backward(grad_wrt_out)
        grad = self.activation.backward(grad)
        grad = self.linear1.backward(grad)
 
        return grad


    def step(self):
        self.linear1.step()
        self.linear2.step()
        

    def zerograd(self):
        pass

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        self.linear1.loadparams(Ws[0],bs[0])
        self.linear2.loadparams(Ws[1],bs[1])
        

    def getWs(self):
        """
        Return the weights for each layer
        Return a list containing weights for first layer then second and so on...
        """
        return (np.array([self.linear1.getW(),self.linear2.getW()]))

    def getbs(self):
        """
        Return the biases for each layer
        Return a list containing bias for first layer then second and so on...
        """
        return (np.array([self.linear1.getb(),self.linear2.getb()]))

class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    def __init__(self, indim, outdim, hiddenlayers=[100,100], alpha=0.1, dropout_probability=0, lr=0.01):
        Transform.__init__(self)
        self.linear1 = LinearMap(indim, outdim, alpha, lr)
        self.activation1 = ReLU(dropout_probability)
        self.linear2 = LinearMap(indim, outdim, alpha, lr)
        self.activation2 = ReLU(dropout_probability)

        self.linear3 = LinearMap(indim, outdim, alpha, lr)

    def forward(self, x, train=True):
        x = self.linear1.forward(x)
        x = self.activation1.forward(x,train)

        x = self.linear2.forward(x)
        x = self.activation2.forward(x)

        x = self.linear3.forward(x)
        

        

        return x

    def backward(self, grad_wrt_out):
        grad = self.linear3.backward(grad_wrt_out)
        grad = self.activation2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.activation1.backward(grad)
        grad = self.linear1.backward(grad)
 
        return grad


    def step(self):
        self.linear1.step()
        self.linear2.step()
        self.linear3.step()

    def zerograd(self):
        pass

    def loadparams(self, Ws, bs):
        self.linear1.loadparams(Ws[0],bs[0])
        self.linear2.loadparams(Ws[1],bs[1])
        self.linear3.loadparams(Ws[2],bs[2])
        

    def getWs(self):
        return (np.array([self.linear1.getW(),self.linear2.getW(),self.linear3.getW()]))

    def getbs(self):
        return (np.array([self.linear1.getb(),self.linear2.getb(),self.linear3.getb()]))


if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    
    with open('omniglot_14.pkl','rb') as f: data = pk.load(f)
    ((trainX,trainY),(testX,testY)) = data
    
    trainY = labels2onehot(trainY)
    testY = labels2onehot(testY)
    
    # show the data
    '''
    plt.plot(trainX[417])
    plt.show()
    plt.plot(trainX[617])
    plt.show()
    plt.plot(trainX[301])
    plt.show()
    plt.plot(trainX[385])
    plt.show()
    '''


    # single layer

    
    
    single1trainloss = []
    single1trainaccuracy = []
    single1testloss = []
    single1testaccuracy = []
    
    def shuffle(X, y, epoch):
        # shuffle function:  work cited from 10-301

        np.random.seed(epoch)
        N = len(y)
        ordering = np.random.permutation(N)
        return X[ordering], y[ordering]

    def train(trainx,trainy,testx,testy, epoch,totalcases,inputsize,nodes,hiddenlayer,alpha, dropout_probability,lr,batch,trainloss,testloss,trainaccu,testaccu):
        # initialize the weights
        w1 = random_weight_init(inputsize,nodes)
        w2 = random_weight_init(nodes,14)
        b1 = zeros_bias_init(nodes)
        b2 = zeros_bias_init(14)
        ws = [w1,w2]
        bs = [b1,b2]
        ann = SingleLayerMLP(inputsize,14,hiddenlayer,alpha,dropout_probability,lr)
        
        ann.loadparams(ws,bs)
        
     
        for i in range (0,epoch):
            tempx = shuffle(trainx, trainy, i)[0]
            tempy = shuffle(trainx, trainy, i)[1]
            soft = SoftmaxCrossEntropyLoss()
            tsoft = SoftmaxCrossEntropyLoss()
            totalmiss = 0
            totaltrainloss = 0
            
            
            
            totalbatch = 0
            for j in range(0,totalcases,batch):
                # get the batch
                input = tempx [j:j+batch,:]
                input = input.T
                
                # get the yhat
                fowardres = ann.forward(input,train= True)
                
                
                # gety
                label = tempy[j:j+batch,:]
                label = label.T
                # find train loss
                totaltrainloss+= soft.forward(fowardres,label)
                totalbatch +=1
                totalmiss += soft.getAccu()
                
                gradout = soft.backward()
                ann.backward(gradout)
                ann.step()
            # accu: (1- totalmiss/totalcases)
            # after iternation
            trainloss.append(totaltrainloss/totalbatch)
           
            trainaccu.append(1- totalmiss/totalcases)


            # get the trained weights bias
            totaltestloss = 0
            testmiss = 0
            testbatch = 0
            ts = SingleLayerMLP(inputsize,14,hiddenlayer,alpha,dropout_probability,lr)
            ts.loadparams(ann.getWs(),ann.getbs())
            for j in range(0,1652,batch):
                # get the batch
                input = testx[j:j+batch,:]
                input = input.T
                
                # get the yhat
                fowardres = ts.forward(input,train= True)
                
                
                # gety
                label = testy[j:j+batch,:]
                label = label.T
                # find train loss
                totaltestloss+= tsoft.forward(fowardres,label)
                testbatch +=1
                testmiss += tsoft.getAccu()
                
                
            testloss.append(totaltestloss/testbatch)
            testaccu.append(1-testmiss/1652)


            
            print(i)

    
            
        
                
    
    epoch =150
    train(trainX,trainY,testX,testY, epoch,6608,11025,50,1,0,0,0.001,32,single1trainloss,single1testloss,single1trainaccuracy,single1testaccuracy)

    print('trainloss')
    print(single1trainloss)
    print('testloss')
    print(single1testloss)
    print('trainaccuracy')
    print(single1trainaccuracy)
    
    print('testaccuracy')
    print(single1testaccuracy)
    
    xlabel = list(range(0,epoch))
    # plot loss
    plt.plot(xlabel,single1trainloss,label = 'train')
    plt.plot(xlabel,single1testloss,label = 'test')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    leg = plt.legend(loc='upper center')
    plt.show()
    # plot accu
    plt.plot(xlabel,single1trainaccuracy,label = 'train')
    plt.plot(xlabel,single1testaccuracy,label = 'test')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    leg = plt.legend(loc='upper center')
    plt.show()

                







        
           


    
    

    pass