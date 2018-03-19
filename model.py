import os
import math
import random

import numpy as np
import tensorflow as tf

from utils import ProgressBar


class MemN2N(object):
    
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.max_words = config.max_words
        self.max_sentences = config.max_sentences
        self.init_mean = config.init_mean
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.anneal_epoch = config.anneal_epoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.max_grad_norm = config.max_grad_norm
        
        self.lin_start = config.lin_start
        self.show_progress = config.show_progress
        self.is_test = config.is_test

        self.checkpoint_dir = config.checkpoint_dir
        
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.query = tf.placeholder(tf.int32, [None, self.max_words], name='input')
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name='time')
        self.target = tf.placeholder(tf.float32, [None, self.nwords], name='target')
        self.context = tf.placeholder(tf.int32, [None, self.mem_size, self.max_words], name='context')
        
        self.hid = []
        
        self.lr = None
        
        if self.lin_start:
            self.current_lr = 0.005
        else:
            self.current_lr = config.init_lr

        self.anneal_rate = config.anneal_rate
        self.loss = None
        self.optim = None
        
        self.sess = sess
        self.log_loss = []
        self.log_perp = []
    
    def build_memory(self):
        self.global_step = tf.Variable(0, name='global_step')
        
        zeros = tf.constant(0, tf.float32, [1, self.edim])
        self.A_ = tf.Variable(tf.random_normal([self.nwords - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        self.B_ = tf.Variable(tf.random_normal([self.nwords - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        self.C_ = tf.Variable(tf.random_normal([self.nwords - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        
        A = tf.concat([zeros, self.A_], axis=0)
        B = tf.concat([zeros, self.B_], axis=0)
        C = tf.concat([zeros, self.C_], axis=0)
        
        self.T_A_ = tf.Variable(tf.random_normal([self.mem_size - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        self.T_C_ = tf.Variable(tf.random_normal([self.mem_size - 1, self.edim], mean=self.init_mean, stddev=self.init_std))
        
        T_A = tf.concat([zeros, self.T_A_], axis=0)
        T_C = tf.concat([zeros, self.T_C_], axis=0)
        
        A_ebd = tf.nn.embedding_lookup(A, self.context)   # [batch_size, mem_size, max_length, edim]
        A_ebd = tf.reduce_sum(A_ebd, axis=2)              # [batch_size, mem_size, edim]
        T_A_ebd = tf.nn.embedding_lookup(T_A, self.time)  # [batch_size, mem_size, edim]
        A_in = tf.add(A_ebd, T_A_ebd)                     # [batch_size, mem_size, edim]
        
        C_ebd = tf.nn.embedding_lookup(C, self.context)   # [batch_size, mem_size, max_length, edim]
        C_ebd = tf.reduce_sum(C_ebd, axis=2)              # [batch_size, mem_size, edim]
        T_C_ebd = tf.nn.embedding_lookup(T_C, self.time)  # [batch_size, mem_size, edim]
        C_in = tf.add(C_ebd, T_C_ebd)                     # [batch_size, mem_size, edim]
        
        query_ebd = tf.nn.embedding_lookup(B, self.query) # [batch_size, max_length, edim]
        query_ebd = tf.reduce_sum(query_ebd, axis=1)      # [batch_size, edim]
        self.hid.append(query_ebd)
        
        for h in range(self.nhop):
            q3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim]) # [batch_size, edim] ==> [batch_size, 1, edim]
            p3dim = tf.matmul(q3dim, A_in, transpose_b=True)     # [batch_size, 1, edim] X [batch_size, edim, mem_size]
            p2dim = tf.reshape(p3dim, [-1, self.mem_size])       # [batch_size, mem_size]
            
            # If linear start, remove softmax layers
            if self.lin_start:
                p = p2dim
            else:
                p = tf.nn.softmax(p2dim)
            
            p3dim = tf.reshape(p, [-1, 1, self.mem_size]) # [batch_size, 1, mem_size]
            o3dim = tf.matmul(p3dim, C_in)                # [batch_size, 1, mem_size] X [batch_size, mem_size, edim]
            o2dim = tf.reshape(o3dim, [-1, self.edim])    # [batch_size, edim]
            
            a = tf.add(o2dim, self.hid[-1]) # [batch_size, edim]
            self.hid.append(a)              # [input, a_1, a_2, ..., a_nhop]
    
