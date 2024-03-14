# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:05:32 2022

@author: tiago
"""
import argparse
import os

def argparser():
    """
    Argument Parser Function
    Outputs:
    - FLAGS: arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='mlm',
        help='standard or mlm')
    
    parser.add_argument(
        '--option',
        type=str,
        default='inhibitors',
        help='unbiased, standard, mlm, experiment1 or experiment2')
    
    parser.add_argument(
        '--receptor',
        type=str,
        default='usp7',
        help='Target receptor (vegfr2,usp7,aa2a,jnk3,drd2 or gsk3)')
    
    parser.add_argument(
        '--normalization_strategy',
        type=str,
        default='percentile',
        help='Predictor normalization strategy (percentile,min-max or robust')
    
    parser.add_argument(
        '--data_path',
        type=dict,
        default={},
        help='Data Path')

    parser.add_argument(
        '--heads_option',
        type=str,
        default='concatenate_heads',
        help='Options: fully_costumize_head_and_layer, concatenate_heads')
    
    parser.add_argument(
        '--layers_options',
        type=str,
        default='all',
        help='Options: single, all')
    
    parser.add_argument(
        '--single_option',
        type=str,
        default='last',
        help='Options: first, last')
    
    parser.add_argument(
        '--computation_strategy',
        type=str,
        default='A',
        help='Options: A,B')

    parser.add_argument(
        '--activation',
        type=str,
        default='none',
        help='Options: none,softmax,tanh,sigmoid')


    parser.add_argument(
        '--mols_to_generate',
        type=int,
        default=200,
        help='Number of molecules to generate')
    
    parser.add_argument(
        '--draw_mols',
        type=bool,
        default=True,
        help='Draw molecules')
    
    parser.add_argument(
        '--show_images',
        type=bool,
        default=False,
        help='Draw molecules')
    
    parser.add_argument(
        '--mols_to_draw',
        type=int,
        default=2,
        help='Number of molecules to generate')

    parser.add_argument(
        '--n_iterations',
        type=int,
        default=200,
        help='Number of iterations')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=7,
        help='Batch size')
    
    parser.add_argument(
        '--plot_attention_scores',
        type=bool,
        default=False,
        help='Print token scores')

    parser.add_argument(
        '--plot_attention_weights',
        type=bool,
        default=False,
        help='Print attention weights matrix')
    
    parser.add_argument(
        '--max_str_len',
        type=int,
        default=100,
        help='SMILES Sequences Max Length')
    
    parser.add_argument(
        '--softmax_activation',
        type=bool,
        default=True,
        help='Apply softmax to the importance weights')
    
    parser.add_argument(
        '--max_stereoisomers',
        type=int,
        default=3,
        help='Top active stereoisomers to select')

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='Optimizer')
    
    parser.add_argument(
        '--vocabulary',
        type=str,
        default='standard',
        help='Options for SMILES vocabulary (standard,steochemistry)')

    parser.add_argument(
        '--seed',
        type=int,
        default=56,
        help='Random seed')
    
    parser.add_argument(
        '--top_tokens_rate',
        type=float,
        default=0.33,
        help='Rate of important tokens ')

    parser.add_argument(
        '--optimizer_fn',
        type=str,
        nargs='+',
        action='append',
        help='Optimizer Function Parameters')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Directory for checkpoint weights'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Directory for log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS


def logging(msg, FLAGS):
    """
    Logging function to update the log file
    Args:
    - msg [str]: info to add to the log file
    - FLAGS: arguments object
    """

    fpath = os.path.join(FLAGS.log_dir, "log.txt")

    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")