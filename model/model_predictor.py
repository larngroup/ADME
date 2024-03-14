# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:27:17 2022

@author: tiago
"""
from utils.utils import Utils
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU
from model.attention import Attention

class Rnn_predictor(tf.keras.Model):
    
    def __init__(self, FLAGS,scaler):
        """ Class for the Predictor

        Args:
            FLAGS (argparse): Implementation parameters
            
        """
        super(Rnn_predictor, self).__init__()
        
        # Parameters' declaration file
        self.FLAGS = FLAGS
        
        # Object to denormalize model predictions
        self.scaler = scaler
        
        # Token vocabulary 
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
          'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
          '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
          '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
        
        # Loading of the training pIC50 labels, used for a type of normalization procedure 
        self.labels = Utils.read_csv(FLAGS)
        
        # Declaration of the model's architecture details
        self.inp_dimension = self.FLAGS.max_str_len
        self.token_len = 47
        self.bidirectional_units = 512
        self.dropout = 0.1
        self.rnn_units = 512
        
        self.bidirectional_layer = tf.keras.layers.Bidirectional(LSTM(self.bidirectional_units, dropout=self.dropout, return_sequences=True)) 
        self.dropout_layer = tf.keras.layers.Dropout(0.1)
        self.rnn_layer = GRU(self.rnn_units, return_sequences=True)
        self.attention_layer = Attention()
        self.dense_layer = tf.keras.layers.Dense(1,activation='linear') 
        

     
    def predict(self, smiles_original):
        """
        This function performs the prediction of the pIC50 for the input 
        molecules
        Parameters
        ----------
        smiles_original: List of SMILES strings to perform the prediction      
        Returns
        -------
        This function performs the denormalization step and returns the model
        prediction
        """
        prediction = self(smiles_original,training=False)

        # prediction_tf = self.predictor(smiles_original,training=False)   
        # print(prediction_tf)          
        # n
        prediction = Utils.denormalization(prediction,self.labels,self.scaler,self.FLAGS)
                
        return prediction
    
    def call(self, sequenc_embed, training=True):

        bidirectional_out = self.bidirectional_layer(sequenc_embed)
        bidirectional_out = self.dropout_layer(bidirectional_out,training=training)
        rnn_out = self.rnn_layer(bidirectional_out)
        rnn_out = self.dropout_layer(rnn_out,training=training)
        attention_out = self.attention_layer(rnn_out)
        pred_out = self.dense_layer(attention_out)
        
        return pred_out
        
    
class Fc_predictor(tf.keras.Model):
    
    def __init__(self, FLAGS,scaler):
        """ Class for the Predictor

        Args:
            FLAGS (argparse): Implementation parameters
            
        """
        super(Fc_predictor, self).__init__()
        
        # Parameters' declaration file
        self.FLAGS = FLAGS
        
        # Object to denormalize model predictions
        self.scaler = scaler
        
        # Token vocabulary 
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
          'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
          '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
          '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
        
        # Loading of the training pIC50 labels, used for a type of normalization procedure 
        self.labels = Utils.read_csv(FLAGS)

        self.FLAGS = FLAGS  
        self.FLAGS.activation_fc = 'relu'    
        
        self.dropout_layer = tf.keras.layers.Dropout(0.1)
        self.final_dense = tf.keras.layers.Dense(1,activation='linear') 
        self.dense_1 = tf.keras.layers.Dense(512,activation=self.FLAGS.activation_fc) 
        self.dense_2 = tf.keras.layers.Dense(256,activation=self.FLAGS.activation_fc) 
        self.dense_3 = tf.keras.layers.Dense(128,activation=self.FLAGS.activation_fc) 

    def predict(self, smiles_original):
        """
        This function performs the prediction of the pIC50 for the input 
        molecules
        Parameters
        ----------
        smiles_original: List of SMILES strings to perform the prediction      
        Returns
        -------
        This function performs the denormalization step and returns the model
        prediction
        """
        prediction = self(smiles_original,training=False)
        pred_np = prediction.numpy()
        # print('prediction: ',pred_np)
        # prediction_tf = self.predictor(smiles_original,training=False)   
        # print(prediction_tf)          
        # n
        prediction = Utils.denormalization(pred_np,self.labels,self.scaler,self.FLAGS)
                        
        return prediction
    
    def call(self, inp, training=True):

        dense_1_out = self.dense_1(inp)
        dense_1_out = self.dropout_layer(dense_1_out,training=training)
        dense_2_out = self.dense_2(dense_1_out)
        dense_2_out = self.dropout_layer(dense_2_out,training=training)
        dense_3_out = self.dense_3(dense_2_out)
        dense_3_out = self.dropout_layer(dense_3_out,training=training)
        
        pred_out = self.final_dense(dense_3_out)
        
        return pred_out
