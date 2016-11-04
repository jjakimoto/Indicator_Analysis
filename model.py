import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import backend as K

class MLP(object):
    """MLP to predict indicator values"""
    def __init__(self, config):
        """Build MLP model through config

        config: the following attributes
            device: device to use for computation, e.g. '/gpu:0'
            save_path: directory to save amd load your model
            is_load: the flag if you use the previous model
            n_stock: the number of caompanies for stock data
            n_batch: batch size for SGD
            n_epochs: the number of interation for training
            learning_rate: initial learning rate value
            anneal: the time when annealing starts
            model: model configuration
        """
        self.device = config['device']
        self.save_path = config['save_path']
        self.is_load = config['is_load']
        self.n_stock = config['n_stock']
        self.n_batch = config['n_batch']
        self.n_epochs = config['n_epochs']
        self.lr = config['learning_rate']
        self.anneal = config['anneal']
        self.model_config = config['model']
        # have compatibility with new tensorflow
        tf.python.control_flow_ops = tf
        # avoid creating _LEARNING_PHASE outside the network
        K.clear_session()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        K.set_session(self.sess)
        with self.sess.as_default():
            with tf.device(self.device):
                self.build_model()
        
    def training(self, input_data, target_data):
        """training MLP with Adam Optimizer
        
        Args:
            input_data (DataFrame): .values have the [None, n_stock] stock prices
            target_data (DataFrame): .values have the [None] indicator values
        """
        # to use for prediciton keep input data
        stock_data = input_data.values
        target_data = target_data.values
        self.data = stock_data
        T = len(stock_data)
        print_freq = int(self.n_epochs / 10)
        if print_freq == 0:
            print_freq = 1
        # index for shuffling
        indices = np.arange(T)
        lr = self.lr
        for epoch in xrange(self.n_epochs):
            np.random.shuffle(indices)
            batch_indices = [indices[i: i + self.n_batch] for i in np.arange(0, T - self.n_batch, self.n_batch)]
            for idx in batch_indices:
                batch_input = [stock_data[t] for t in idx]
                batch_target = [target_data[t] for t in idx]
                # select transition from pool
                self.sess.run(self.optim, 
                              feed_dict={self.input_: batch_input,
                                         self.target: batch_target,
                                         self.learning_rate: lr,
                                         K.learning_phase(): 1})  
            # update learning rate by annealing
            lr = lr * self.anneal / max(self.anneal, epoch)
            
            
            if epoch % print_freq == 0:
                print ("epoch:",  epoch)
                loss = self.sess.run(self.loss, feed_dict={self.input_: stock_data,
                                                           self.target: target_data,
                                                          K.learning_phase(): 0})
                print ('loss_test:', loss)
            
            
        print ("finished training")
        
    def predict(self, input_data):
        """Predict indicator values

        Args:
            input_data (DataFrame): .values have the [None, n_stock] stock prices
        """    
        # to stabilize we use more than self.n_batch
        index = input_data.index
        data = input_data.values
        n_data = len(data)
        if n_data < self.n_batch:
            n_add = self.n_batch - n_data 
            data = np.concatenate((self.data[-n_add:], data))
        prediction = self.sess.run(self.output, feed_dict={self.input_: data, 
                                                            K.learning_phase(): 0})[-n_data:]
        return pd.DataFrame(prediction, index=index)

    def accuracy(self, input_data, target_data):
        target_value = target_data.values
        prediction = self.predict(input_data).values
        return np.mean((target_value - prediction)**2)

    
    def build_model(self):
        """Build the network and optimizations """
        self.network = self.build_network()
        self.input_ =  tf.placeholder(tf.float32, [None, self.n_stock], name='input')
        self.target = tf.placeholder(tf.float32, [None], name='target')
        network_output = self.network(self.input_)
        self.output = tf.reduce_sum(network_output * self.input_, 1)
        self.loss = tf.reduce_mean((self.output - self.target)**2)
        # optimization
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.optim = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss)
        tf.initialize_all_variables().run(session=self.sess)
        # save model
        self.saver = tf.train.Saver()
            
        
    def build_network(self):
        """Build MLP network"""
        model = Sequential()
        model.add(Lambda(lambda x: x,  input_shape=(self.n_stock,)))
        for conf in self.model_config:
            if conf['is_drop']:
                model.add(Dropout(conf['drop_rate']))
            model.add(Dense(conf['n_hidden']))
            if conf['is_batch']:
                model.add(BatchNormalization(mode=1, axis=-1))
            model.add(PReLU())
        model.add(Dense(self.n_stock))
        return model
    
    
    def save(self):
        """Save model at self.save_path"""
        save_path = self.saver.save(self.sess, self.save_path)
        print("Model saved in file: %s" % save_path)

    
    def load(self):
        """Load model from self.save_path if possible"""
        print(" [*] Reading checkpoints...")
        try:
            self.saver.restore(self.sess, self.save_path)
            return True
        except:
            return False
