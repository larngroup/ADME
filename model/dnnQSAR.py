# -*- coding: utf-8 -*-
# Internal 
from utils.utils import Utils
from model.model_predictor import Rnn_predictor,Fc_predictor

# External 
import tensorflow as tf
import numpy as np
import joblib


       
class DnnQSAR_model:
    
    def __init__(self,FLAGS):
        """
        Initializes and loads biological affinity Predictor
        """
   
        self.FLAGS = FLAGS
        
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
          'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
          '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
          '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
        
        self.labels = Utils.read_csv(FLAGS)
        
        if FLAGS.receptor == 'usp7':
            self.scaler = joblib.load('data//usp7_scaler.save') 
            filepath_mlm = 'models//predictor//predictor_mlm_usp7.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'aa2a':
            self.scaler = joblib.load('data//aa2a_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_aa2a.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'drd2':
            self.scaler = joblib.load('data//drd2_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_drd2.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'jnk3':
            self.scaler = joblib.load('data//jnk3_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_jnk3.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'gsk3':
            self.scaler = joblib.load('data//gsk3_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_gsk3.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'vgfr2':
            self.scaler = joblib.load('data//vgfr2_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_vgfr2.h5'
            filepath_standard = 'models//predictor//predictor_standard_vgfr2.h5' 
        
        if FLAGS.option == 'mlm' or FLAGS.option == 'mlm_exp1' or FLAGS.option == 'mlm_exp2'  or FLAGS.option == 'unbiased':
            self.predictor = Fc_predictor(FLAGS)
            sequence_in = tf.constant([list(np.ones((1,256)))])
            prediction_test = self.predictor(sequence_in)
            self.predictor.load_weights(filepath_mlm)
        

        elif FLAGS.option == 'standard' or FLAGS.option == 'standard_exp1' or FLAGS.option == 'standard_exp2':
            self.predictor = Rnn_predictor(FLAGS)
            sequence_in = tf.constant([list(np.ones((1,256)))])
            prediction_test = self.predictor(sequence_in)
            self.predictor.load_weights(filepath_standard)

    
    def predict(self, smiles_original):
        """
        This function performs the prediction of the USP7 pIC50 for the input 
        molecules
        Parameters
        ----------
        smiles_original: List of SMILES strings to perform the prediction      
        Returns
        -------
        This function performs the denormalization step and returns the model
        prediction
        """
        prediction = self.predictor.predict(smiles_original)
        # print(prediction)
        # prediction_tf = self.predictor(smiles_original,training=False)   
        # print(prediction_tf)          
        # n
        prediction = Utils.denormalization(prediction,self.labels,self.scaler)
                
        return prediction
        
