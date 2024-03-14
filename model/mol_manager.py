    # -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from dataloader.dataloader import DataLoader
from utils.utils import Utils  
from model.transformer_mlm import Transformer_mlm

# external
import tensorflow as tf
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw,Descriptors,QED,Crippen
from rdkit.Chem.IFG import ifg
import warnings
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)

class Mol_manager:
    """Conditional Generation Object"""
    
    def __init__(self, FLAGS):
        
        # Parameters' file
        self.FLAGS = FLAGS
        
        if self.FLAGS.vocabulary == 'standard':
            # Loading the selected SMILES vocabulary
            self.token_table = Utils().standard_voc 
        elif self.FLAGS.vocabulary == 'stereo':
            # Loading the selected SMILES vocabulary
            self.token_table = Utils().stereo_voc 
        
        self.special_tokens = Utils().special_tokens
        
        # Building the token-integer dictionary  
        self.tokenDict = Utils.smilesDict(self.token_table)
        
        # Load pre-trained models (Generators, Predictor and Transformer-encoder)
        self.generator_unbiased = DataLoader().load_generator(self.FLAGS,'unbiased')
    
        self.generator_biased = DataLoader().load_generator(self.FLAGS,'biased')
        
        if FLAGS.model == 'mlm':
            self.predictor = DataLoader().load_predictor(self.FLAGS)
            self.transformer_model = Transformer_mlm(self.FLAGS)
       

    def load_process_data(self):   
        """ 
        Loads the set of molecules to analyse, tokenizes the sequences and 
        identifies the FG and NFG. 
        Returns
        -------
        The outcome is saved on a dataframe. 
        """
        if self.FLAGS.option == 'generated':
            smiles = DataLoader().load_promising_mols(self.FLAGS,'unbiased')
        elif self.FLAGS.option == 'inhibitors':
            smiles = DataLoader().load_promising_mols(self.FLAGS,'inhibitors')
            print('Number of inhibitors: ', len(smiles))
            
        # Transform SMILES into mols
        mols = [Chem.MolFromSmiles(seq) for seq in smiles] 
        
        # Identify indexes of FG and NFG atoms
        self.df = pd.DataFrame()
        fgs_all = []
        nfgs_all = []
        tokens_all = []
        for i,mol in enumerate(mols):
            mol_smi = smiles[i]
            print(i,mol_smi)
            smiles[i] = mol_smi
            nfg_idxs = []
            fgs = ifg.identify_functional_groups(mol)
            fg_idxs = []
            for fg in fgs:
                fg_idxs = fg_idxs + list(fg[0])
                
            for atom in mol.GetAtoms():
                if atom.GetIdx() not in fg_idxs:
                    nfg_idxs.append(atom.GetIdx())
                    
            # Tokenization to update indexes based on other tokens (ring tokens)
            N = len(mol_smi)
            i = 0
            j= 0
            tokens = []
            while (i < N):
                for j in range(len(self.token_table)):
                    symbol = self.token_table[j]
                    if symbol == mol_smi[i:i + len(symbol)]:
                        tokens.append(symbol)
                        # print(symbol)
                        i += len(symbol)
                        break   
       
            idx_atom = -1
            idx_real = -1 
            fg_idxs_updated = []
            nfg_idxs_updated = []
            for t in tokens:
                idx_real +=1
                if t not in self.special_tokens:
                    idx_atom +=1
                    if idx_atom in fg_idxs:
                        fg_idxs_updated.append(idx_real)
                    elif idx_atom in nfg_idxs:
                        nfg_idxs_updated.append(idx_real)

            fgs_all.append(fg_idxs_updated)
            nfgs_all.append(nfg_idxs_updated)
            tokens_all.append(tokens)
        
        self.df['smiles'] = smiles
        self.df['fgs'] = fgs_all
        self.df['nfgs'] = nfgs_all
        self.df['tokens'] = tokens_all
        
    def compute_aw(self):  
        """
        Transformer calculates the attention weights and the atom importance 
        score (IS), according to the specifications in the argument_parser.py file. 
        Then, it is computed the average IS for FG and NFG atoms and based on
        that calculation the function calculates the two evaluation metrics.
        
        Returns
        -------
        The rate of molecules for which the average IS_fg is higher than IS_nfg
        and the Mann-Whitney statistical test.
        """
        
        imp_fg_all = np.zeros((1,len(self.df['smiles'])))
        imp_nfg_all = np.zeros((1,len(self.df['smiles'])))
        smiles = self.df['smiles']
        
        # Apply the Multi-Head Attention to extract token scores and contextual embeddings
        for i,smi in enumerate(smiles):
            token_importance,tokens_transf = self.transformer_model.process_mols(smi)
            
            fg_idxs = self.df['fgs'].iloc[i]
            nfg_idxs = self.df['nfgs'].iloc[i]
            tokens_general = self.df['tokens'].iloc[i]
            
            # Extract token importance for FG and NFG atoms
            imp_fg = token_importance[fg_idxs]
            imp_nfg = token_importance[nfg_idxs]
            
            # Compute the averages
            mean_fg = np.mean(imp_fg)
            mean_nfg = np.mean(imp_nfg)
            
            imp_fg_all[0,i] = mean_fg
            imp_nfg_all[0,i] = mean_nfg
                
        # Plot the scatter plot with the averages IS for each molecule
        Utils.plot_scatter(imp_fg_all,imp_nfg_all)
        print('\nRate of molecules where FG att > NFG att: ', ((imp_fg_all > imp_nfg_all).sum())/len(imp_nfg_all[0,:]))
        
        # Application of the Mann-Whitney U test
        # H0: the distributions of both samples are equal.
        stat, p = mannwhitneyu(imp_fg_all[0,:].tolist(), imp_nfg_all[0,:].tolist())
        print('\nP-value Mann-Whitney: ',p)
        print('\nstat=%.2f, p=%.30f' % (stat, p))
        if p > 0.05:
        	print('Probably the same distribution')
        else:
            print('Probably different distributions')
            
        Utils.print_parameters(self.FLAGS)

        
    def samples_generation(self,smiles_data=None):
        """
        Generates new hits, computes the desired properties, draws the 
        specified number of hits and saves the set of optimized compounds to 
        a file.
        
        Parameters:
        -----------
        smiles_data (DataFrame): Set of generated molecules and corresponding 
                                 properties
    
        """
        
 
        training_data = DataLoader().load_generator_smiles(self.FLAGS.models_path['generator_data_path'])

        mols_id = 'biased'
        if smiles_data is None:
            print('\n*********** Sampling of new molecules ('+self.FLAGS.option+ ' model) ***********')
            if self.FLAGS.option == 'unbiased':
                generator = self.generator_unbiased
                mols_id = 'unbiased'
                save_path =  self.FLAGS.models_path['sampled_unbiased_path']
            else:
                print('biased model loaded')
                generator = DataLoader().load_generator(self.FLAGS,'biased_checkpoint')
                save_path = self.FLAGS.models_path['sampled_biased_path']

            smiles_data = Utils().generate_smiles(generator,self.predictor,self.transformer_model,self.tokenDict,self.FLAGS)
        else:
            save_path = self.FLAGS.models_path['sampled_biased_path']
            
        vld = (len(smiles_data['smiles'])/self.FLAGS.mols_to_generate)*100
                
        tanimoto_int = Utils().external_diversity(list(smiles_data['smiles']))
        print('\nInternal Tanimoto Diversity: ',round(tanimoto_int,4))
        
        tanimoto_ext = Utils().external_diversity(list(smiles_data['smiles']),list(training_data))
        print('\nExternal Tanimoto Diversity: ',round(tanimoto_ext,4))
    
        Utils().plot_hist(smiles_data['sas'],self.FLAGS.mols_to_generate,"sas")
        Utils().plot_hist(smiles_data['pIC50 usp7'],self.FLAGS.mols_to_generate,"usp7")
        
        qed_values = []
        logp_values = []
        
        with open(save_path, 'w') as f:
            f.write("Number of molecules: %s\n" % str(len(smiles_data['smiles'])))
            f.write("Percentage of valid and unique molecules: %s\n\n" % str(vld))
            f.write("SMILES, pIC50, SAS, MW, logP, QED\n")
            for i,smi in enumerate(smiles_data['smiles']):
                mol = list(smiles_data['mols_obj'])[i]
                q = QED.qed(mol)
                mw, logP = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
                
                qed_values.append(q)
                logp_values.append(logP)
                data = str(list(smiles_data['smiles'])[i]) + " ," +  str(np.round(smiles_data['pIC50 usp7'][i],2)) + " ," + str(np.round(smiles_data['sas'][i],2)) + " ,"  + str(np.round(mw,2)) + " ," + str(np.round(logP,2)) + " ," + str(np.round(q,2))
                f.write("%s\n" % data)  
                
        smiles_data['QED'] = qed_values
        smiles_data['LogP'] = logp_values
        
        success_mols = smiles_data[(smiles_data['pIC50 usp7']>=6) & (smiles_data['sas']<4.0)  & (smiles_data['LogP']>-1) & (smiles_data['LogP']<5)]

        print('\nSuccess_rate for all molecules (%): ', round((len(list(set(success_mols['smiles'])))/self.FLAGS.mols_to_generate)*100,4))
        print('\nSuccess_rate for valid molecules (%): ', round((len(list(set(success_mols['smiles'])))/len(list(set(smiles_data['smiles']))))*100,4))
        
        print("\nMax QED: ", round(np.max(qed_values),4))
        print("Mean QED: ", round(np.mean(qed_values),4))
        print("Std QED: ", round(np.std(qed_values),4))
        print("Min QED: ", round(np.min(qed_values),4))
    
        print("\nMax logP: ", round(np.max(logp_values),4))
        print("Mean logP: ", round(np.mean(logp_values),4))
        print("Std logP: ", round(np.std(logp_values),4))
        print("Min logP: ", round(np.min(logp_values),4))   

        if self.FLAGS.draw_mols == True:
            df_sorted = smiles_data.sort_values('pIC50 usp7',ascending = False)
            self.drawMols(list(df_sorted['smiles'])[:self.FLAGS.mols_to_draw])
            
        
    def drawMols(self,smiles_generated=None):
        """
        Function that draws the chemical structure of given compounds

        Parameters:
        -----------
        smiles_generated (list): It contains a set of generated molecules
        
        Returns
        -------
        Returns a figure with the 2D structure of the specified number of molecules
        """
        # Definition of the drawing options
        DrawingOptions.atomLabelFontSize = 50
        DrawingOptions.dotsPerAngstrom = 100
        DrawingOptions.bondLineWidth = 3
        DrawingOptions.addStereoAnnotation = True  
        DrawingOptions.addAtomIndices = True
        
        
        if smiles_generated is None:
            smiles_generated = ['CC(C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12',
                                  'Cc1cccc(C)c1C(=O)NC1CCCNC1=O','CC1CCC(C)C12C(=O)Nc1ccccc12',
                                  'NC(Cc1ccc(O)cc1)C(=O)O','CC(=O)OCC(=O)C1(OC(C)=O)CCC2C3CCC4=CC(=O)CCC4(C)C3CCC21C',
                                  'CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1',
                                  'CC1(C)CCC(C)(C)c2cc(C(=O)Nc3ccc(C(=O)O)cc3)ccc21','CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1',
                                  'CCC(c1ccccc1)c1ccc(OCCN(CC)CC)cc1','Cc1cccc(C)c1NC(=O)CN(C)C(=O)C(C)C','CC(C)C(CO)Nc1ccnc2cc(Cl)ccc12',
                                  'Cc1ccc(C(=O)Nc2ccc(C(C)C)cc2)cc1Nc1nccc(-c2ccc(C(F)(F)F)cc2)n1','CN(C)CCCNC(=O)CCC(=O)Nc1ccccc1',
                                  'CC(C)CC(=O)N(Cc1ccccc1)C1CCN(Cc2ccccc2)CC1']
            
            known_drugs = ['C[C@@H]1[C@H]2C3=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@@]4(C)[C@]3(C)CC[C@@]2(C(=O)O)CC[C@H]1C',
                           'O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1','CCC1(c2ccc(N)cc2)CCC(=O)NC1=O',
                           'CC(N)(Cc1ccc(O)cc1)C(=O)O','CC(=O)O[C@]1(C(C)=O)CC[C@H]2[C@@H]3C=C(C)C4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]21C',
                           'COc1c(N2C[C@@H]3CCCN[C@@H]3C2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12',
                           'C=C(c1ccc(C(=O)O)cc1)c1cc2c(cc1C)C(C)(C)CCC2(C)C','O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O',
                           'CCN(CC)CCOc1ccc(Cc2ccccc2)cc1', 'CCN(CC)CC(=O)Nc1c(C)cccc1C', 'CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12', 
                           'Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1','O=C(CCCCCCC(=O)Nc1ccccc1)NO',
                           'CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1']
            
            legends = ['Ursolic acid', 'Thalidomide', 'Aminoglutethimide',
                       'Racemetyrosine', 'Megestrol acetate', 'Moxifloxacin',
                       'Bexarotene', 'Ciproflaxicin', 'Tesmilifene', 'Lidocaine', 
                       'Hydroxycloroquine', 'Nilotilib', 'Vorinostat', 'Fentanyl']
            
           
            drugs_mols = Utils().smiles2mol(known_drugs)
       
            img = Draw.MolsToGridImage(drugs_mols, molsPerRow=3, subImgSize=(300,300),legends=legends)
            img.show()
            
        generated_mols = Utils().smiles2mol(smiles_generated)
            
        img = Draw.MolsToGridImage(generated_mols, molsPerRow=3, subImgSize=(300,300))
        img.show()