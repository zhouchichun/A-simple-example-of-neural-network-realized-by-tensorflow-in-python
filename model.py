#!/usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

'''
The model can be created as a class (as you can see this treatment will bring us alot of benifits). 
Inside the model there should be method related to inputs, the method layer, the optimizer, the loss, and the train.
Moreover methods related to the logging, the checkpoint file, and the pb file should also be included in the model class.
'''

class model(object):
    def __init__(self,sess,config,logging):
        self.sess=sess
        self.config=config
        self.logging=logging
        
        
        self._checkpoint_path=self.config["ckpt"]
        
        self.global_step=tf.Variable(0,trainable=False)
        self.build()
        self.print_var()

        self.loggingAll()
        self._saver=tf.train.Saver(tf.global_variables(),max_to_keep=10)
        self.initialize()
    def loggingAll(self):
        for name in dir(self):
            if name.find("_")==0 and name.find("__")==-1:
                self.logging.info("self.%s\t%s"%(name,str(getattr(self,name))))
        
    def readckpt(self):
        ckpt=tf.train.get_checkpoint_state(self._checkpoint_path)
        if ckpt:
            self.logging.info("reading training record from '%s'"%ckpt.model_checkpoint_path)
            self._saver.restore(self.sess,ckpt.model_checkpoint_path)      
            return True
        return False
        
    def _input(self):
        self.x=tf.placeholder(tf.float32,[None],name="x")
        self.y=tf.placeholder(tf.float32,[None],name="y")
    def _initial_linear(self):
        with tf.variable_scope("linear"):
            w=tf.Variable(1.0,name="weight_linear")
            b=tf.Variable(0.0,name="bias_linear")
        self.out_put=tf.multiply(w,self.x)+b
    def nonlinear_sigmoid(self):
        with tf.variable_scope("non_linear_sigmoid"):
            self.out_put=tf.sigmoid(self.out_put)
    def linear(self):
        with tf.variable_scope("linear"):
            w=tf.Variable(1.0,name="weight_linear")
            b=tf.Variable(0.0,name="bias_linear")
        self.out_put=tf.multiply(w,self.out_put)+b
    def nonlinear_tanh(self):
        with tf.variable_scope("non_linear_tanh"):
            self.out_put=tf.tanh(self.out_put)
    def loss(self):
        with tf.variable_scope("loss"):
            w=tf.Variable(1.0,name="weight_linear")
            b=tf.Variable(0.0,name="bias_linear")
        self.predict=tf.multiply(w,self.out_put)+b
        self.loss=tf.nn.l2_loss(self.predict-self.y)
    def print_var(self):
        for item in dir(self):
            type_string=str(type(getattr(self,item)))
            print(item,type_string)
        
    def opt(self):
        self._opt=tf.train.AdamOptimizer(self.config["learn_rate"])
        self._train_opt=self._opt.minimize(self.loss,global_step=self.global_step)
        
    def build(self):
        self._input()
        self._initial_linear()
        self.nonlinear_sigmoid()
        self.linear()
        self.nonlinear_tanh()
        self.linear()
        self.nonlinear_sigmoid()
        self.loss()
        self.opt()
        self.logging.info("model is built!")
    def initialize(self):
        if not self.readckpt():
            self.sess.run(tf.global_variables_initializer())
    def train(self,input_x,input_y,i):
        feed_dict={self.x:input_x,self.y:input_y}
        loss,global_step,_=self.sess.run([self.loss,self.global_step,self._train_opt],feed_dict)
        if i%10==0:
            print ("loss is %s,global_step is %s,i is %s"%(loss,global_step,i))
            self.logging.info("loss is %s,global_step is %s,i is %s"%(loss,global_step,i))
            if i%500==0:
                predict=self.sess.run(self.predict,feed_dict)
                plt.plot(predict[90:100])
                plt.plot(input_y[90:100])
                plt.show()
                
                if i%10000==0:
                    self._saver.save(self.sess,self._checkpoint_path+"checkpoint",global_step=global_step)
                    predict=self.sess.run(self.predict,feed_dict)
                    plt.plot(predict[0:100])
                    plt.plot(input_y[0:100])
                    plt.show()
