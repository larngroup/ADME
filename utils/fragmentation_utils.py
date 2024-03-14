# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 08:58:58 2023

@author: tiago
"""
# from global_parameters import 70
from rdkit import Chem
import numpy as np

# Atom numbers of noble gases (should not be used as dummy atoms)
NOBLE_GASES = {2, 10, 18, 36, 54, 86}
ng_correction = set()

# Divide a molecule into fragments
def split_molecule(mol):

    split_id = 70

    res = []
    to_check = [mol]
    while len(to_check) > 0:
        ms = spf(to_check.pop(), split_id)
        if len(ms) == 1:
            res += ms
        else:
            to_check += ms
            split_id += 1

    return create_chain(res)


# Function for doing all the nitty-gritty splitting work.
def spf(mol, split_id):

    bonds = mol.GetBonds()
    for i in range(len(bonds)):
        if okToBreak(bonds[i]):
            mol = Chem.FragmentOnBonds(mol, [i], addDummies=True, dummyLabels=[(0, 0)])
            # Dummy atoms are always added last
            n_at = mol.GetNumAtoms()
            mol.GetAtomWithIdx(n_at-1).SetAtomicNum(split_id)
            mol.GetAtomWithIdx(n_at-2).SetAtomicNum(split_id)
            return Chem.rdmolops.GetMolFrags(mol, asMols=True)

    # If the molecule could not been split, return original molecule
    return [mol]


# Check if it is ok to break a bond.
# It is ok to break a bond if:
#    1. It is a single bond
#    2. Either the start or the end atom is in a ring, but not both of them.
def okToBreak(bond):

    if bond.IsInRing():
        return False

    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False

    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    if not(begin_atom.IsInRing() or end_atom.IsInRing()):
        return False
    elif begin_atom.GetAtomicNum() >= 70 or \
            end_atom.GetAtomicNum() >= 70:
        return False
    else:
        return True


# Build up a chain of fragments from a molecule.
# This is required so that a given list of fragments can be rebuilt into the same
#   molecule as was given when splitting the molecule
def create_chain(splits):
    splits_ids = np.asarray(
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= 70]) for m in splits])

    splits_ids = \
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= 70]) for m in splits]

    splits2 = []
    mv = np.max(splits_ids)
    look_for = [mv if isinstance(mv, np.int64) else mv[0]]
    join_order = []
    mols = []

    for i in range(len(splits_ids)):
        l = splits_ids[i]
        if l[0] == look_for[0] and len(l) == 1:
            mols.append(splits[i])
            splits2.append(splits_ids[i])
            splits_ids[i] = []


    while len(look_for) > 0:
        sid = look_for.pop()
        join_order.append(sid)
        next_mol = [i for i in range(len(splits_ids))
                      if sid in splits_ids[i]]

        if len(next_mol) == 0:
            break
        next_mol = next_mol[0]

        for n in splits_ids[next_mol]:
            if n != sid:
                look_for.append(n)
        mols.append(splits[next_mol])
        splits2.append(splits_ids[next_mol])
        splits_ids[next_mol] = []

    return [simplify_splits(mols[i], splits2[i], join_order) for i in range(len(mols))]


# Split and keep track of the order on how to rebuild the molecule
def simplify_splits(mol, splits, join_order):

    td = {}
    n = 0
    for i in splits:
        for j in join_order:
            if i == j:
                td[i] = 70 + n
                n += 1
                if n in NOBLE_GASES:
                    n += 1

    for a in mol.GetAtoms():
        k = a.GetAtomicNum()
        if k in td:
            a.SetAtomicNum(td[k])

    return mol
