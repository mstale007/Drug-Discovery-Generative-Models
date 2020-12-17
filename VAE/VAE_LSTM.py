import keras as K
from Dataloader.smile_tokenizer import SmilesTokenizer
from keras.layers import Input,Dense, Lambda
from tensorflow.keras.layers import GRU
from keras.models import Model
from tensorflow.keras import regularizers
import numpy as np

def sample_z(args):
    mu, log_sigma,latent_space_dim = args
    eps = K.backend.random_normal(shape=(latent_space_dim, ), mean=0., stddev=1.)
    return mu + K.backend.exp(log_sigma / 2.) * eps
    
class VAE(object):
  def __init__(self):
    self.log_sigma=0
    self.mu=0
    self.latent_space_dim=10
    self.mini_batch_size=10
    self.vae,self.encoder,self.decoder=self.createModel()
  
  def createModel(self):
    smile_tokenizer=SmilesTokenizer()
    vector_len = len(smile_tokenizer.table)

    #Encoder
    _input=Input(shape=(None,vector_len,))
    hidden=Dense(256,activation='linear')(_input)
    hidden=GRU(128,activation='relu',return_sequences=True,dropout=0.3,kernel_initializer='ones',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(hidden)
    hidden=GRU(64,activation='relu',return_sequences=True,dropout=0.2)(hidden)
    self.mu = Dense(self.latent_space_dim, activation='linear')(hidden)
    self.log_sigma = Dense(self.latent_space_dim, activation='linear')(hidden)
    z = Lambda(sample_z)([self.mu, self.log_sigma,self.latent_space_dim])
    #z = Lambda(sample_z([self.mu, self.log_sigma,self.latent_space_dim]))

    #Decoder
    decoder_hidden1=GRU(64,activation='relu',return_sequences=True,
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))
    decoder_hidden2=GRU(128,activation='relu',return_sequences=True,dropout=0.2)
    decoder_hidden3=Dense(256,activation='sigmoid')
    decoder_hidden4=Dense(vector_len,activation='sigmoid')

    h_p=decoder_hidden1(z)
    h_p=decoder_hidden2(h_p)
    h_p=decoder_hidden3(h_p)
    _output=decoder_hidden4(h_p)

    #VAE
    vae=Model(_input,_output)

    #Encoder Part
    encoder=Model(_input,self.mu)

    #Decoder Part
    d_in = Input(shape=(None,self.latent_space_dim))
    d_h = decoder_hidden1(d_in)
    d_h = decoder_hidden2(d_h)
    d_h = decoder_hidden3(d_h)
    d_out = decoder_hidden4(d_h)
    decoder=Model(d_in,d_out)

    return vae,encoder,decoder

  def vae_loss(self,y_true, y_pred):
      """ Calculate loss = reconstruction loss + KL loss for eatch data in minibatch """
      # E[log P(X|z)]
      recon = K.sum(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))
      recon*=256
      # D_KL(Q(z|X) || P(z|X)); calculate in closed from as both dist. are Gaussian
      kl = 0.5 * K.sum(K.sum(K.exp(self.log_sigma) + K.square(self.mu) - 1. - self.log_sigma, axis=1))
      return recon + kl
  
#   def sample_z(self,mu,log_sigma):
#     #mu, log_sigma = args
#     eps = K.backend.random_normal(shape=(self.mini_batch_size, self.latent_space_dim), mean=0., stddev=1.)
#     temp=np.array(log_sigma )/ 2.
#     return mu + K.exp(temp) * eps