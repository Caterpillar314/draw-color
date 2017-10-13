import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os
import argparse
import csv
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Draw():
    def __init__(self, args, conf):

        self.img_size = 64
        self.img_initial_size = 64
        self.num_colors = 3

        self.attention = conf['attention']
        self.attention_n = 5
        self.read = self.read_attention if self.attention else self.read_basic
        self.write = self.write_attention if self.attention else self.write_basic

        self.n_hidden = conf['n_hidden']
        self.n_z = conf['nz_dim']
        self.sequence_length = conf['sequence_length']
        self.batch_size = 64
        self.share_parameters = False

        self.dataset = args.dataset
        self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])

        self.e = tf.random_normal((self.batch_size, self.n_z), mean=0, stddev=1) # Qsampler noise

        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        self.cs = [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        x = tf.reshape(self.images, [-1, self.img_size*self.img_size*self.num_colors])
        self.attn_params = []
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.img_size * self.img_size * self.num_colors)) if t == 0 else self.cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            r = self.read(x,x_hat,h_dec_prev)
            # encode it to gauss distrib
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

        self.generation_loss = tf.nn.l2_loss(x - self.generated_images)

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


    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x,x_hat], 1)

    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)
        # we have the parameters for a patch of gaussian filters. apply them.
        def filter_img(img, Fx, Fy, gamma):
            # Fx,Fy = [64,5,32]
            # img = [64, 32*32*3]

            img = tf.reshape(img, [-1, self.img_size, self.img_size, self.num_colors])
            img_t = tf.transpose(img, perm=[3,0,1,2])

            # color1, color2, color3, color1, color2, color3, etc.
            batch_colors_array = tf.reshape(img_t, [self.num_colors * self.batch_size, self.img_size, self.img_size])
            Fx_array = tf.concat([Fx, Fx, Fx], 0)
            Fy_array = tf.concat([Fy, Fy, Fy], 0)

            Fxt = tf.transpose(Fx_array, perm=[0,2,1])

            # Apply the gaussian patches:
            glimpse = tf.matmul(Fy_array, tf.matmul(batch_colors_array, Fxt))
            glimpse = tf.reshape(glimpse, [self.num_colors, self.batch_size, self.attention_n, self.attention_n])
            glimpse = tf.transpose(glimpse, [1,2,3,0])
            glimpse = tf.reshape(glimpse, [self.batch_size, self.attention_n*self.attention_n*self.num_colors])
            # finally scale this glimpse w/ the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])
        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        return tf.concat([x, x_hat],1)

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

    def generate(self, args, batch_size=64, nb_batch=100, save_imgs=False) :

        print('Creating Dataset...')

        data = glob("../dataset/"+self.dataset+"/*")
        processed_data = [get_image(f, self.img_initial_size, is_crop=True) for f in data[0:batch_size*nb_batch]]

        print('Restoring network...')

        path = args.folder+"/dataviz/"
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/"+args.folder+"/checkpoints/"))

        print('Processing data...')

        X = np.zeros((batch_size * nb_batch, self.n_z), dtype=np.float32)
        with open(path+'results.csv', 'w', newline='') as csvfile :
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['dim'+str(k) for k in range(self.n_z)])

            for i in range(nb_batch):
                print('Batch : '+str(i))
                batch = processed_data[i*self.batch_size:(i+1)*self.batch_size]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2

                mu = self.sess.run(self.mu, feed_dict={self.images: batch_images})

                for j in range(batch_size):
                    writer.writerow(mu[-1][j].tolist())
                    X[i*self.batch_size+j] = mu[-1][j].tolist()

                if save_imgs :
                    if not os.path.exists(path+str(i)):
                        os.makedirs(path+str(i))
                    for j in range(batch_size):
                        ims(path+str(i)+'/img'+str(j)+'.jpg', processed_data[i*self.batch_size+j])
        X_embedded = TSNE().fit_transform(X)
        print(X_embedded.shape)

        plt.scatter(X_embedded[:,0], X_embedded[:,1])
        plt.savefig(args.folder+"/dataviz/data_viz.png")


def bool_arg(string):
    value = string.lower()
    if value == 'true': return True
    elif value == 'false': return False
    else: raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='logs/CelebA/', type=str, help="Folder where is stored the training checkpoints", dest="folder")
    parser.add_argument('-d', '--dataset', default='CelebA', type=str, help="Which dataset to use", dest="dataset")
    parser.add_argument('-s', '--save_imgs', default=False, type=bool_arg, help="Whether to save the images or not", dest="save_imgs")
    parser.add_argument('-n', '--nb_batch', default=10, type=int, help="Number of batches to analyse", dest="nb_batch")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    with open(args.folder+'args.json', 'r') as d :
        conf = json.load(d)

    model = Draw(args, conf)
    model.generate(args, nb_batch=args.nb_batch, save_imgs=args.save_imgs)
