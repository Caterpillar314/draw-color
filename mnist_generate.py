import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import input_data
import argparse
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json


class Draw():
    def __init__(self, conf):
        self.mnist = input_data.read_data_sets("../dataset/mnist/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.img_size = 28
        self.attention = conf['attention']
        self.attention_n = 5
        self.read = self.read_attention if self.attention else self.read_basic
        self.write = self.write_attention if self.attention else self.write_basic

        self.n_hidden = conf['n_hidden']
        self.n_z = conf['nz_dim']
        self.sequence_length = conf['sequence_length']
        self.batch_size = 64
        self.nb_steps = conf['nb_steps']
        self.share_parameters = False

        self.images = tf.placeholder(tf.float32, [None, 784])
        self.e = tf.random_normal((self.batch_size, self.n_z), mean=0, stddev=1) # Qsampler noise
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        self.cs = [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        x = self.images
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.img_size**2)) if t == 0 else self.cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            r = self.read(x,x_hat,h_dec_prev)
            # encode it to guass distrib
            self.mu[t], self.logsigma[t], self.sigma[t], enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1))
            # sample from the distrib to get z
            z = self.sampleQ(self.mu[t],self.sigma[t])
            # retrieve the hidden layer of RNN
            h_dec, dec_state = self.decode_layer(dec_state, z)
            # map from hidden layer -> image portion, and then write it.
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
            self.share_parameters = True # from now on, share variables

        # the final timestep
        self.generated_images = tf.nn.sigmoid(self.cs[-1])

        self.generation_loss = tf.reduce_mean(-tf.reduce_sum(self.images * tf.log(1e-10 + self.generated_images) + (1-self.images) * tf.log(1e-10 + 1 - self.generated_images),1))

        kl_terms = [0]*self.sequence_length
        for t in range(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t]
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - self.sequence_length*0.5
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
        self.cost = self.generation_loss + self.latent_loss
        optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g,5),v)
        self.train_op = optimizer.apply_gradients(grads)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    # given a hidden decoder layer:
    # locate where to put attention filters
    def attn_window(self, scope, h_dec):
        with tf.variable_scope(scope, reuse=self.share_parameters):
            parameters = dense(h_dec, self.n_hidden, 5)
        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters,5,1)

        # move gx/gy to be a scale of -imgsize to +imgsize
        gx = (self.img_size+1)/2 * (gx_ + 1)
        gy = (self.img_size+1)/2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # stride/delta: how far apart these patches will be
        delta = (self.img_size - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        # returns [Fx, Fy, gamma]
        return self.filterbank(gx,gy,sigma2,delta) + (tf.exp(log_gamma),)

    # Given a center, distance, and spread
    # Construct [attention_n x attention_n] patches of gaussian filters
    # represented by Fx = horizontal gaussian, Fy = vertical guassian
    def filterbank(self, gx, gy, sigma2, delta):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32),[1, -1])
        # centers for the individual patches
        mu_x = gx + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_y = gy + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im = tf.reshape(tf.cast(tf.range(self.img_size), tf.float32), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        Fy = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        # normalize so area-under-curve = 1
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),1e-8)
        return Fx, Fy


    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x,x_hat], 1)

    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)
        # we have the parameters for a patch of gaussian filters. apply them.
        def filter_img(img, Fx, Fy, gamma):
            Fxt = tf.transpose(Fx, perm=[0,2,1])
            img = tf.reshape(img, [-1, self.img_size, self.img_size])
            # Apply the gaussian patches:
            # keep in mind: horiz = imgsize = verts (they are all the image size)
            # keep in mind: attn = height/length of attention patches
            # allfilters = [attn, vert] * [imgsize,imgsize] * [horiz, attn]
            # we have batches, so the full batch_matmul equation looks like:
            # [1, 1, vert] * [batchsize,imgsize,imgsize] * [1, horiz, 1]
            glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
            glimpse = tf.reshape(glimpse, [-1, self.attention_n**2])
            # finally scale this glimpse w/ the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])
        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        return tf.concat([x, x_hat], 1)

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder",reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_enc(image, prev_state)

        # map the RNN hidden state to latent variables
        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = dense(hidden_layer, self.n_hidden, self.n_z)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            logsigma = dense(hidden_layer, self.n_hidden, self.n_z)
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, next_state


    def sampleQ(self, mu, sigma):
        return mu + sigma*self.e

    def decode_layer(self, prev_state, latent):
        # update decoder RNN with latent var
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_dec(latent, prev_state)

        return hidden_layer, next_state

    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        with tf.variable_scope("write", reuse=self.share_parameters):
            decoded_image_portion = dense(hidden_layer, self.n_hidden, self.img_size**2)
        return decoded_image_portion

    def write_attention(self, hidden_layer):
        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = dense(hidden_layer, self.n_hidden, self.attention_n**2)
        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)
        Fyt = tf.transpose(Fy, perm=[0,2,1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.matmul(Fyt, tf.matmul(w, Fx))
        wr = tf.reshape(wr, [self.batch_size, self.img_size**2])
        return wr * tf.reshape(1.0/gamma, [-1, 1])

    def generate(self, args, batch_size=64) :
        print('Started generating...')

        path = args.folder+"/results/"
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/"+args.folder+"/checkpoints/"))

        h_dec_prev = tf.zeros((batch_size, self.n_hidden))
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        for t in range(self.sequence_length) :
            c_prev = tf.zeros((batch_size, self.img_size**2)) if t == 0 else self.cs[t-1]
            mean_array = 2*args.max_mean*np.random.rand(batch_size, self.n_z) - args.max_mean
            z_array = np.random.normal(mean_array, args.std)
            z = tf.convert_to_tensor(z_array, dtype=tf.float32)
            h_dec, dec_state = self.decode_layer(dec_state, z)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

        cs = self.sess.run(self.cs)
        cs = 1.0/(1.0+np.exp(-np.array(cs)))
        for cs_iter in range(10):
            results = cs[cs_iter]
            results_square = np.reshape(results, [-1, 28, 28])
            ims(path+"gen-step-"+str(cs_iter)+"-mmean"+str(args.max_mean)+"-std"+str(args.std)+".jpg",merge(results_square,[8,8]))


def bool_arg(string):
    value = string.lower()
    if value == 'true': return True
    elif value == 'false': return False
    else: raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='logs/CelebA/', type=str, help="Folder where is stored the training checkpoints", dest="folder")
    parser.add_argument('-s', '--std', default=1., type=float, help="Standard deviation for generating the latent vector", dest="std")
    parser.add_argument('-m', '--max_mean', default=0., type=float, help="Maximum mean for latent vector", dest="max_mean")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    with open(args.folder+'args.json', 'r') as d :
        conf = json.load(d)

    model = Draw(conf)
    model.generate(args)
