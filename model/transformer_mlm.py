# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from utils.utils import Utils
from model.transformers import Masked_Smiles_Model

# external
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

class Transformer_mlm:
    """Transformer general Class"""
    def __init__(self, FLAGS):
        
        # Parameters' declaration file
        self.FLAGS = FLAGS
        
        # Loading the selected SMILES vocabulary
        if self.FLAGS.vocabulary == 'standard':
            self.token_table = Utils().standard_voc 
        elif self.FLAGS.vocabulary == 'stereo':
            self.token_table = Utils().stereo_voc 
             
        self.vocab_size = len(self.token_table)
        
        # Dictionary that makes the token-integer correspondence
        self.tokenDict = Utils.smilesDict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Model parameters' definition
        self.max_length = 150
        self.model_size = 256
        self.n_heads = 4
        self.n_layers = 4
        self.dropout = 0.1
        self.activation_func = 'relu'
        self.ff_dim = 1024
        
        # Compute positional encoding
        self.pes = []
        for i in range(self.max_length):
            self.pes.append(Utils.positional_encoding(i, self.model_size ))
        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)
        
        # Initializes and loads the parameters of the pre-trained model
        self.build_models()
    
    def build_models(self):
        """ Builds the Transformer architecture"""
        
        self.encoder = Masked_Smiles_Model(self.model_size,self.ff_dim,self.n_heads,self.n_layers,
                                           self.max_length,self.vocab_size,self.activation_func)     
        
        sequence_in = tf.constant(np.zeros((1,150)))
        _,_,all_weights,_ = self.encoder(sequence_in)
        
        if self.FLAGS.vocabulary == 'standard':
            self.encoder.load_weights(self.FLAGS.models_path['transformer_mlm_standard'])  
        elif self.FLAGS.vocabulary == 'stereo':
            self.encoder.load_weights(self.FLAGS.models_path['transformer_mlm_standard'])  
            
         
        
    
    def loss_function(self,y_true,y_pred, mask):
        """ Calculates the loss (sparse categorical crossentropy) just for 
            the masked tokens

        Args
        ----------
            y_true (array): true label
            y_pred (array): model's predictions
            mask (array): mask of 1's (masked tokens) and 0's (rest)
            
        Returns
        -------
           loss
        """
        loss_ = self.crossentropy(y_true,y_pred)
        mask = tf.cast(mask,dtype=loss_.dtype)
        loss_ *= mask
    
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

        
    def accuracy_function(self, y_true,y_pred, mask):
        """ Calculates the accuracy of masked tokens that were well predicted 
            by the model

        Args
        ----------
            y_true (array): true label
            y_pred (array): model's predictions
            mask (array): mask of 1's (masked tokens) and 0's (rest)
            
        Returns
        -------
           accuracy
        """
        accuracies = tf.equal(y_true, tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.float64))
        mask_b = tf.constant(mask > 0)
        accuracies = tf.math.logical_and(mask_b, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32) 
        mask = tf.cast(mask, dtype=tf.float32) 
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
    def masking(self, inp, fgs, threshold_min):
        """ Performs the masking of the input smiles. 20% of the tokens are 
            masked (10% from FG and 10% from other regions of the molecule)

        Args
        ----------
            inp (list): List of input mols
            fgs (list): List with indices of FG
            threshold_min (int): minimum number of FGs where masking considers 
                                 FGs'
            
        Returns
        -------
            x (list): SMILES with masked tokens
            masked_positions (list): Sequences of the same length of the smiles
                                     with 1's on the masked positions
        """
        
        masked_positions = []
        x = [i.copy() for i in inp]
        
        for smile_index in range(len(x)):
            fg = fgs[smile_index]
            not_fg = [indx for indx in range(len(x[smile_index])) if (indx not in fg) and (x[smile_index][indx] not in [self.token_table.index('<CLS>'),
			self.token_table.index('<PAD>'), self.token_table.index('<SEP>'), self.token_table.index('<MASK>') ])] 
            
            # from the 20% of tokens that will be masked, half will be from the fg and the other half from the rest of the schleton
            p_fg = 0.1
            p_not_fg = 0.1 
            
            if len(fg) < threshold_min: 
                p_not_fg = 0.15
                
            num_mask_fg = max(1, int(round(len(fg) * p_fg))) 
            num_mask_not_fg = max(1, int(round(len(not_fg) * p_not_fg))) 
            shuffle_fg = random.sample(range(len(fg)), len(fg)) 
            shuffle_not_fg = random.sample(range(len(not_fg)), len(not_fg))
			
            fg_temp = [fg[n] for n in shuffle_fg[:num_mask_fg]] 
            not_fg_temp = [not_fg[n] for n in shuffle_not_fg[:num_mask_not_fg]] 
			
            mask_index = fg_temp + not_fg_temp
            masked_pos =[0]*len(x[smile_index])
			
            for pos in mask_index:
                masked_pos[pos] = 1
                if random.random() < 0.8: 
                    x[smile_index][pos] = self.token_table.index('<MASK>')
                elif random.random() < 0.15: 
                    index = random.randint(1, self.token_table.index('<CLS>')-1) 
                    x[smile_index][pos] = index
            masked_positions.append(masked_pos) 
                    
        return x, masked_positions
        

    def process_mols(self,sequence_in):
        """
        Parameters
        ----------
        sequence_in (str): Input SMILES sequence
        FLAGS (argparse): Parameter declaration file
        
        Returns
        -------
        tokens_importance: attention scores
        

        """
                
        # Tokenize - transform the SMILES strings into lists of tokens 
        tokens,smiles_filtered, token_len = Utils.tokenize_and_pad(self.FLAGS,[sequence_in],self.token_table,'mlm')   
     
        # Identify the functional groups of each molecule
        fgs = Utils.identify_fg(smiles_filtered)
                 
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_raw = Utils.smiles2idx(tokens,self.tokenDict)
                     
        # 'Minimumm number of functional groups (FGs) where masking considers FGs'
        threshold_min = 6
         
        # Masking of the source sequence
        self.input_masked, self.masked_positions = self.masking(input_raw, fgs, threshold_min)   
        
        # Transformer encoder: maps the input tokens to the contextual embeddings and outputs attention weights
        _,_,alignments,aw_layers_heads = self.encoder(tf.constant( self.input_masked),training=False)
        # alignments 4layers(1,150,150) - heads concatenated
        # aw_layers_heads 4layers(4,150,150) - all heads
        
        # Attention weight's processing according to the selections in the parameters file
        if self.FLAGS.heads_option == 'fully_costumize_head_and_layer':
            # customize the the desired strategy
            selected_layer_idx = 0 # from 0 to 3
            selected_head_idx = 3 # from 0 to 3
            plot_attentionhead = False
            
            attention_scores_all_heads = aw_layers_heads[selected_layer_idx].numpy()
            selected_head = attention_scores_all_heads[selected_head_idx,:token_len,:token_len] 
       
            if plot_attentionhead:
                            
                ## plot all attention weights (n,n)
                size_seq = len(selected_head)   
                
                ax = plt.gca()
                ax.matshow(selected_head)
                ax.set_xticks(range(size_seq))
                ax.set_yticks(range(size_seq))
               
                labels_tokens = [token for token in tokens[0][:token_len]]
                try:
                    ax.set_xticklabels(
                    labels_tokens, rotation=90)
                except:
                    print('size mismatch')
               
                ax.set_yticklabels(labels_tokens)
                
            # Extract the importance of each specific token
            importance_all = []
            
            size_h = len(selected_head)
            for c in range(0,size_h):
    
                importance_element = []
                importance_element.append(selected_head[c,c])
                
                for v in range(0,size_h):
                    if v!=c:
                        element = (selected_head[c,v] + selected_head[v,c])/2
                        importance_element.append(element)
            
                importance_all.append(importance_element)
            
    
            importance_tokens = [np.mean(l) for l in importance_all]
            
        
        elif self.FLAGS.heads_option == 'concatenate_heads':
            if self.FLAGS.layers_options == 'single' and self.FLAGS.computation_strategy == 'A':
                # last/first layer, all heads, raw attention values

                # Extract only the last layer 
                attention_scores = alignments[self.FLAGS.single_option].numpy()
                selected_attention = attention_scores[0,:token_len,:token_len]
                 
                # Extract the importance of each specific token
                importance_all = []            
                size_h = len(selected_attention)
                for c in range(0,size_h):
        
                    importance_element = []
                    importance_element.append(selected_attention[c,c])
                    
                    for v in range(0,size_h):
                        if v!=c:
                            importance_element.append(selected_attention[c,v])
                            importance_element.append(selected_attention[v,c])
                
                    importance_all.append(importance_element)
                importance_tokens = [np.mean(l) for l in importance_all]
         
            
            elif self.FLAGS.layers_options == 'single' and self.FLAGS.computation_strategy == 'B':            
                 # Last layer, all heads, average attention values
                 # Extract only the last layer 
                 attention_scores = alignments[-1].numpy()
                 selected_attention = attention_scores[0,:token_len,:token_len]
         
                 # Extract the importance of each specific token
                 importance_all = []            
                 size_h = len(selected_attention)
                 for c in range(0,size_h):
                     importance_element = []
                     importance_element.append(selected_attention[c,c])
                    
                     for v in range(0,size_h):
                        if v!=c:
                            element = (selected_attention[c,v] + selected_attention[v,c])/2
                            importance_element.append(element)
                        
                     importance_all.append(importance_element)
                 importance_tokens = [np.mean(l) for l in importance_all]
        
            elif self.FLAGS.layers_options == 'all' and self.FLAGS.computation_strategy == 'A':
            
                # Extract values from all layers
                alignments_np = [align.numpy() for align in alignments]
                alignments_layers = np.zeros((1,self.max_length,self.max_length))
                for l in range(len(alignments_np)):
                    alignments_layers = alignments_layers + alignments_np[l][0,:,:]
                
                selected_attention = alignments_layers[0,:token_len,:token_len]
                
                # Extract the importance of each specific token
                importance_all = []
              
                size_h = len(selected_attention)
                for c in range(0,size_h):
        
                    importance_element = []
                    importance_element.append(selected_attention[c,c])
                    
                    for v in range(0,size_h):
                        if v!=c:
                            importance_element.append(selected_attention[c,v])
                            importance_element.append(selected_attention[v,c])
                
                    importance_all.append(importance_element)
         
                importance_tokens = [np.mean(l) for l in importance_all]
            
            elif self.FLAGS.layers_options == 'all' and self.FLAGS.computation_strategy == 'B': 
                # Extract values from all layers
                alignments_np = [align.numpy() for align in alignments]
                alignments_layers = np.zeros((1,self.max_length,self.max_length))
                for l in range(len(alignments_np)):
                    alignments_layers = alignments_layers + alignments_np[l][0,:,:]
                
     
                selected_attention = alignments_layers[0,:token_len,:token_len]
                
                # Extract the importance of each specific token
                importance_all = []
              
                size_h = len(selected_attention)
                for c in range(0,size_h):
        
                    importance_element = []
                    importance_element.append(selected_attention[c,c])
                    
                    for v in range(0,size_h):
                        if v!=c:
                            element = (selected_attention[c,v] + selected_attention[v,c])/2
                            importance_element.append(element)
                            
                    importance_all.append(importance_element)
         
                importance_tokens = [np.mean(l) for l in importance_all]
         
                  
        if self.FLAGS.plot_attention_scores:
            scores = Utils.softmax(importance_tokens)
            
            # Sort keeping indexes
            sorted_idxs = np.argsort(-scores)
                        
            # Identify most important tokens
            number_tokens = int((self.FLAGS.top_tokens_rate)*len(sequence_in))
            
            # Define the attention score threshold above which the important tokens are considered
            threshold = scores[sorted_idxs[number_tokens]]
        
            # Plot the important tokens        
            plt.figure(figsize=(15,7))
            plt.axhline(y = threshold, color = 'r', linestyle = '-')
            plt.plot(scores,linestyle='dashed')
            ax = plt.gca()
            ax.set_xticks(range(len(sequence_in)))
            ax.set_xticklabels(sequence_in)
            plt.xlabel('Sequence')
            plt.ylabel('Attention weights')
            plt.show()
        
        if self.FLAGS.plot_attention_weights:
            
            size_seq = len(selected_attention)   
            fig = plt.figure(figsize=(20,15))
            ax = plt.gca()
            fontdict = {'fontsize': 15}
            im = ax.matshow(selected_attention)
            ax.set_xticks(range(size_seq))
            ax.set_yticks(range(size_seq))
           
            labels_tokens = [token for token in tokens[0][:token_len]]
            try:
                ax.set_xticklabels(
                labels_tokens, fontdict=fontdict,rotation=90)
            except:
                print('size mismatch')
            ax.set_yticklabels(labels_tokens,fontdict=fontdict)
            fig.colorbar(im, fraction=0.046, pad=0.04)
            
            
        
        importance_tokens = Utils.apply_activation(importance_tokens,self.FLAGS)
        
        return np.array(importance_tokens[1:]), tokens 


        