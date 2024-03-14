# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:05:42 2021

@author: tiago
"""

# internal
from model.mol_manager import Mol_manager
from model.argument_parser import argparser,logging

# External
import tensorflow as tf
import warnings
import time
import os

warnings.filterwarnings('ignore')


if __name__ == '__main__':
        
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
          
    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    
    FLAGS.models_path = {'generator_unbiased_path': 'models//generator//unbiased_generator.hdf5',
                       'generator_biased_path': 'models//generator//biased_generator.hdf5',
                       'transformer_mlm_standard': 'models//transformer//model_standard.h5',
                       'transformer_mlm_stereo': 'models//transformer//model_stereo.h5',
                       'predictor_standard': 'models//predictor//predictor_standard.h5',
                       'predictor_mlm': 'models//predictor//predictor_mlm.h5',
                       'generator_data_path': 'data/example_data.smi',
                       'most_promising_hits': 'generated/best_hits.smi',
                       'sampled_unbiased_path': 'generated/unbiased_set.smi',
                       'sampled_biased_path': 'generated/biased_set.smi',
                       'known_inhibitors': 'data/known_inhibitors.csv',
                       'predictor_data_path': 'data/data_vegf2.csv'} 
    
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

        
    logging(str(FLAGS), FLAGS)
    
    # Initialization of the class
    mol_manager = Mol_manager(FLAGS)
    
    # Loading of the most important of the class
    mol_manager.load_process_data()
    
    # Computation of the token importance and interpretation of the results 
    mol_manager.compute_aw()
