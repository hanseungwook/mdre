import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.keras.initializers as initializers
from tensorflow.keras import regularizers
import numpy as np
import seaborn as sns; 
import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.stats import norm, uniform, cauchy
from scipy.linalg import block_diag, inv, det
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
slim = tf.contrib.slim
from tqdm.notebook import tqdm
from time import sleep
from IPython.display import display, clear_output
import pickle
import os

tf.keras.backend.set_floatx('float32')


# tol = 1e-35
# do = 0.8
# mi=80


def reset(seed=40):
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)
    


def ratios_critic(x, K, l1,l2,input_dim):
    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        W_psd=[0]*K
        mus=[0]*K
        b=[0]*K
        h=[0]*K
#         x = slim.fully_connected(x, l2, activation_fn=tf.nn.leaky_relu)
        for i in range(K):
        
            W_psd[i] = tf.get_variable('W'+str(i),(l2,l2),initializer=tf.keras.initializers.Identity()) 
            W_psd[i] = tf.matmul(W_psd[i],W_psd[i],transpose_b=True)
            mus[i] = tf.get_variable('mu'+str(i),(l2),initializer=tf.constant_initializer(0.)) #
            b[i] = tf.get_variable('b'+str(i),(1),initializer=tf.constant_initializer(-320)) 

            x_ = tf.expand_dims(x-mus[i],-1)

            h[i] = tf.matmul(x_,W_psd[i],transpose_a=True)
            h[i] = tf.matmul(h[i],x_)
            h[i] = tf.squeeze(-0.5*h[i] + b[i],-1) #+ slim.fully_connected(x, 1, activation_fn=None,  biases_initializer=None) 
#             if i==2 or i==K-1:
#                 h[i] = -tf.nn.relu(h[i]) #- 550
#             else:
#             h[i] = tf.nn.relu(-h[i]) - 500
        
        h = tf.convert_to_tensor(h)
        return tf.squeeze(tf.transpose(h)) #150 450 200 [-80.-40,-160.-100]


def get_gt_ratio_kl(p,q,samples):
    ratio = p.log_prob(samples) - q.log_prob(samples)
    kl = tf.reduce_mean(ratio)
    return ratio, kl

def get_logits(samples, K, n_dims):
#     samples = tf.expand_dims(samples,1)
    return ratios_critic(samples, K, n_dims, n_dims, n_dims)

def get_kl_from_cob(samples_p, samples_q, K, n_dims):
    log_rat = get_logits(samples_p, K, n_dims)
    V_p = log_rat[:,0]-log_rat[:,1]
    return tf.reduce_mean(V_p)

def get_loss(samples,bs=500,K=5,n_dims=320):
    logits_=[0]*len(samples)
    identity=np.eye(len(samples))
    labels_=[0]*len(samples)
    disc_loss=0
    for i,sample in enumerate(samples):
        logits_[i] = get_logits(sample,K,n_dims)
        labels_[i] = tf.reshape(np.tile(identity[i],bs),[bs,K])
        disc_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_[i], labels=labels_[i]))
    
    return disc_loss 


def get_optim(loss, lr=0.001, b1=0.001, b2=0.999, steps=1500, alpha=0.0):
    t_vars = tf.trainable_variables()
    c_vars = [var for var in t_vars if 'critic' in var.name]
    '''Uncomment if not using cosine annealing'''
#     optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2).minimize(loss, var_list=t_vars)
#     optim = tf.train.AdamOptimizer(lr).minimize(loss, var_list=t_vars)
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.cosine_decay(lr, global_step, steps, alpha=alpha, name=None) #4000, 6500, 20000
    # Passing global_step to minimize() will increment it at each step.
    optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    return optim 

    
    
    