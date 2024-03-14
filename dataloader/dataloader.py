# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""
# Internal
from model.generator import Generator 
from model.model_predictor import Rnn_predictor,Fc_predictor 

# External
from keras.models import Sequential
import csv
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
import joblib
import numpy as np
import tensorflow as tf

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_generator(FLAGS,generator_type):
        """ Initializes and loads the weights of the trained generator, 
            whether the unbiased or biased model.

        Args
        ----------
            FLAGS (argparse): Model parameters
            generator_type (str): Indication of the model to load: (biased, 
                                biased from last checkpoint or unbiased)
        Returns
        -------
            generator_model (sequential): model with the pre-trained weights
        """
                
        generator_model = Sequential()
        generator_model=Generator(FLAGS,True)
        
        path = ''
        if generator_type == 'biased':
            path =FLAGS.models_path['generator_biased_path'] 
        elif generator_type == 'biased_checkpoint':
            path = FLAGS.checkpoint_path+'biased_generator.hdf5'
        elif generator_type == 'unbiased':
            path =FLAGS.models_path['generator_unbiased_path'] 
        
        generator_model.model.load_weights(path)
        
        return generator_model
    
    
    @staticmethod
    def load_predictor(FLAGS):
        """ Loads the evaluation the pre-trained Predictor models and 
        corresponding scaler to denormalize predictions
        Args
        ----------
            FLAGS (argparse): Parameters file
        
        Returns
        -------
            predictor (model): The pre-trained Predictor
        """
                
        if FLAGS.receptor == 'usp7':
            scaler = joblib.load('data//usp7_scaler.save') 
            filepath_mlm = 'models//predictor//predictor_mlm_usp7.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'aa2a':
            scaler = joblib.load('data//aa2a_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_aa2a.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'drd2':
            scaler = joblib.load('data//drd2_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_drd2.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'jnk3':
            scaler = joblib.load('data//jnk3_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_jnk3.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'gsk3':
            scaler = joblib.load('data//gsk3_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_gsk3.h5'
            filepath_standard = 'models//predictor//predictor_standard_usp7.h5' 
        elif FLAGS.receptor == 'vgfr2':
            scaler = joblib.load('data//vgfr2_scaler.save')
            filepath_mlm = 'models//predictor//predictor_mlm_vgfr2.h5'
            filepath_standard = 'models//predictor//predictor_standard_vgfr2.h5' 
        
        if FLAGS.model == 'mlm' or FLAGS.model == 'mlm_exp1' or FLAGS.model == 'mlm_exp2'  or FLAGS.model == 'unbiased':
            
            predictor = Fc_predictor(FLAGS,scaler)
            sequence_in = tf.constant([list(np.ones((1,256)))])
            prediction_test = predictor(sequence_in)
            predictor.load_weights(filepath_mlm)
        
        elif FLAGS.model == 'standard' or FLAGS.model == 'standard_exp1' or FLAGS.model == 'standard_exp2':
            predictor = Rnn_predictor(FLAGS,scaler)
            sequence_in = tf.constant([list(np.ones((1,256)))])
            prediction_test = predictor(sequence_in)
            predictor.load_weights(filepath_standard)
        
        return predictor
    
    
    @staticmethod
    def load_promising_mols(FLAGS,model='biased'):
        """ Loads a set of pre-generated molecules

        Args
        ----------
            FLAGS (argparse): Parameter declaration file

        Returns
        -------
            smiles_list (list): Set of promising hits previously generated
        """
        
        smiles = []
        pic50_values = []
        mw_values = []
        sas_values = []
        logp_values = []
        qed_values = []
        tpsa_values = []
        h_donors_values = []
        h_acceptors_values = []
        rotablebonds_values = []
        n_rings = []
        file_ids =[]
        df = pd.DataFrame()
          
        if model == 'filtered':
             
            with open(FLAGS.model_paths['most_promising_hits'], 'r') as csvFile:
                reader = csv.reader(csvFile)  
                it = iter(reader) 
                for idx,row in enumerate(it):
                        
                    try:
                        m = Chem.MolFromSmiles(row[0])
                        s = Chem.MolToSmiles(m)
                        if s not in smiles:
                            smiles.append(s)
                    except:
                        next
            return smiles
            
        elif model == 'inhibitors':
            
            # Load the known inhibitors from CHEmbl dataset
            with open(FLAGS.models_path['predictor_data_path']) as f:
                lines = f.readlines()
                for idx,line in enumerate(lines):
                    if idx > 0  :
                        line_i = line.split(';')    
                        try:
                            smiles_i = line_i[7]
                            pic50_i =  line_i[12]
                            if  len(smiles_i)<=149 and smiles_i not in smiles and float(pic50_i) > 6.5 and '[Na+]' not in smiles_i and '/' not in smiles_i and '[2H]' not in smiles_i and '[Si]' not in smiles_i and '[S@@]' not in smiles_i and '[o+]' not in smiles_i and '@' not in smiles_i:
                                mol = Chem.MolFromSmiles(smiles_i,sanitize=True)
                                s = Chem.MolToSmiles(mol)
                                smiles.append(smiles_i)    
                        except:
                            next
 
            # Load the known inhibitors
            with open(FLAGS.models_path['known_inhibitors']) as f:
                lines = f.readlines()
                for idx,line in enumerate(lines):
                    # print(idx)
                    if idx > 0  :
                        line = line.split(' ')
                        
                        try:
                            seq = line[2]
                            if seq not in smiles:
                                mol = Chem.MolFromSmiles(seq)
                                s = Chem.MolToSmiles(mol)
                                smiles.append(s)
                        except:
                            print(line)
                            next
            return smiles
                            
        else:    
            if model == 'unbiased':
                paths_old_pred = [FLAGS.models_path['sampled_unbiased_path']]
            else:
                # paths_old_pred = ["generated/sample_mols_biased_512_final_all.smi","generated/sample_mols_biased_256_final.smi","generated/sample_mols_biased_final_512_normal.smi"]
                paths_old_pred = [FLAGS.models_path['sampled_biased_path']]
            for fp_id,fp in enumerate(paths_old_pred):
                with open(fp, 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    
                    it = iter(reader)
                    # next(it, None)  # skip first item.    
                    for idx,row in enumerate(it):
                            
                        try:
                            m = Chem.MolFromSmiles(row[0])
                            s = Chem.MolToSmiles(m)
                            if s not in smiles:
                                smiles.append(s)
                                # print(fp_id)
                 
                                pic50_values.append(float(row[1])) 
                                sas_values.append(float(row[2]))
                                mw_values.append(float(row[3]))
                                logp_values.append(float(row[4]))
                                qed_values.append(float(row[5]))
                                file_ids.append(fp_id)
                                        
                            
                                tpsa_values.append(round(Descriptors.TPSA(m),2))
                                h_donors_values.append(Chem.Lipinski.NumHDonors(m))
                                h_acceptors_values.append(Chem.Lipinski.NumHAcceptors(m))
                                rotablebonds_values.append(Chem.Lipinski.NumRotatableBonds(m))
                                n_rings.append(Chem.Lipinski.RingCount(m))
                        
                        except:
                            next
                
           
            df['smiles'] = smiles
            df['pic50'] = pic50_values
            df['sas'] = sas_values
            df['mw'] = mw_values
            df['logp'] = logp_values
            df['qed'] = qed_values
            df['tpsa'] = tpsa_values 
            df['hdonors'] = h_donors_values
            df['hacceptors'] = h_acceptors_values 
            df['rotable_bonds'] = rotablebonds_values
            df['n_rings'] = n_rings 
            df['file_id']  = file_ids
    
            return df

    @staticmethod
    def load_generator_smiles(data_path):
        """ Loads the molecular dataset and filters the compounds considered
             syntactically invalid by RDKit.
        Args:
            data_path (str): The path of all SMILES set

        Returns:
            all_mols (list): The list with the training and testing SMILES of
                               the Transformer model
        """
       
        all_mols = [] 
        file = open(data_path,"r")
        lines = file.readlines()        
        for line in lines:
            x = line.split()
            try:
                mol = Chem.MolFromSmiles(x[0].strip())
                smi = Chem.MolToSmiles(mol)
                all_mols.append(smi)
            except:
                print("Invalid molecule")
        
        return all_mols
    
    
        