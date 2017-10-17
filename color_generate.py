import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os
import argparse
import json

class Draw():
    def __init__(self, conf):

        self.img_size = 64
        self.img_initial_size = 128
        self.num_colors = 3

        self.attention = conf['attention']
        self.attention_n = 5
        self.write = self.write_attention if self.attention else self.write_basic

        self.n_hidden = conf['n_hidden']
        self.n_z = conf['nz_dim']
        self.sequence_length = conf['sequence_length']
        self.batch_size = 64
        self.share_parameters = False

        self.dataset = conf['dataset']

        self.e = tf.random_normal((self.batch_size, self.n_z), mean=0, stddev=1) # Qsampler noise

        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        self.cs = [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        #x = tf.reshape(self.images, [-1, self.img_size*self.img_size*self.num_colors])
        self.attn_params = []
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.img_size * self.img_size * self.num_colors)) if t == 0 else self.cs[t-1]
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

        #self.generation_loss = tf.nn.l2_loss(x - self.generated_images)

        kl_terms = [0]*self.sequence_length
        for t in range(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t]

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

        self.attn_params.append([gx, gy, delta])

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
            decoded_image_portion = dense(hidden_layer, self.n_hidden, self.img_size*self.img_size*self.num_colors)
            # decoded_image_portion = tf.reshape(decoded_image_portion, [-1, self.img_size, self.img_size, self.num_colors])
        return decoded_image_portion

    def write_attention(self, hidden_layer):
        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = dense(hidden_layer, self.n_hidden, self.attention_n*self.attention_n*self.num_colors)

        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n, self.num_colors])
        w_t = tf.transpose(w, perm=[3,0,1,2])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)

        # color1, color2, color3, color1, color2, color3, etc.
        w_array = tf.reshape(w_t, [self.num_colors * self.batch_size, self.attention_n, self.attention_n])
        Fx_array = tf.concat([Fx, Fx, Fx], 0)
        Fy_array = tf.concat([Fy, Fy, Fy], 0)

        Fyt = tf.transpose(Fy_array, perm=[0,2,1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.matmul(Fyt, tf.matmul(w_array, Fx_array))
        sep_colors = tf.reshape(wr, [self.batch_size, self.num_colors, self.img_size**2])
        wr = tf.reshape(wr, [self.num_colors, self.batch_size, self.img_size, self.img_size])
        wr = tf.transpose(wr, [1,2,3,0])
        wr = tf.reshape(wr, [self.batch_size, self.img_size * self.img_size * self.num_colors])
        return wr * tf.reshape(1.0/gamma, [-1, 1])

    def generate(self, args, batch_size=64) :
        print('Started generating...')

        path = args.folder+"/generation/"
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/"+args.folder+"/checkpoints/"))

        h_dec_prev = tf.zeros((batch_size, self.n_hidden))
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        for t in range(self.sequence_length) :
            c_prev = tf.zeros((self.batch_size, self.img_size * self.img_size * self.num_colors)) if t == 0 else self.cs[t-1]
            mean_array = 2*args.max_mean*np.random.rand(batch_size, self.n_z) - args.max_mean
            z_array = np.random.normal(mean_array, args.std)
            z = tf.convert_to_tensor(z_array, dtype=tf.float32)
            h_dec, dec_state = self.decode_layer(dec_state, z)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

        cs, attn_params = self.sess.run([self.cs, self.attn_params])
        cs = 1.0/(1.0+np.exp(-np.array(cs)))

        for cs_iter in range(self.sequence_length):
            results = cs[cs_iter]
            results_square = np.reshape(results, [-1, self.img_size, self.img_size, self.num_colors])
            ims(path+"gen-step-"+str(cs_iter)+"-mmean"+str(args.max_mean)+"-std"+str(args.std)+".jpg",merge_color(results_square,[8,8]))

            for i in range(64):
                center_x = int(attn_params[cs_iter][0][i][0])
                center_y = int(attn_params[cs_iter][1][i][0])
                distance = int(attn_params[cs_iter][2][i][0])

                size = 2;

                for x in range(3):
                    for y in range(3):
                        nx = x - 1;
                        ny = y - 1;

                        xpos = center_x + nx*distance
                        ypos = center_y + ny*distance

                        xpos2 = min(max(0, xpos + size), 63)
                        ypos2 = min(max(0, ypos + size), 63)

                        xpos = min(max(0, xpos), 63)
                        ypos = min(max(0, ypos), 63)

                        results_square[i,xpos:xpos2,ypos:ypos2,0] = 0;
                        results_square[i,xpos:xpos2,ypos:ypos2,1] = 1;
                        results_square[i,xpos:xpos2,ypos:ypos2,2] = 0;


            ims(path+"/gen-clean-step-"+str(cs_iter)+".jpg",merge_color(results_square,[8,8]))


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
