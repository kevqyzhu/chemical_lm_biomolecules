from rdkit import Chem
from rdkit.Chem import Descriptors
import random
import selfies as sf
from matplotlib import pyplot as plt
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import re
import copy
import pandas as pd
from io import BytesIO
from PIL import Image

import group_selfies
from group_selfies import (
        fragment_mols, 
        Group, 
        MolecularGraph, 
        GroupGrammar, 
        group_encoder
    )

def DrawMolsZoomed(mols, legends, molsPerRow=3, subImgSize=(300, 300)):#, leg): #https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html#rdkit.Chem.Draw.rdMolDraw2D.MolDraw2D
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow: nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])

    full_image = Image.new('RGB', fullSize )
    for ii, mol in enumerate(mols):
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        le = legends[ii]
        column = ii % molsPerRow
        row = ii // molsPerRow
        offset = ( column*subImgSize[0], row * subImgSize[1] )
        d2d = rdMolDraw2D.MolDraw2DCairo(subImgSize[0], subImgSize[1])
        d2d.DrawMolecule(mol,legend=le)
        d2d.FinishDrawing()
        sub = Image.open(BytesIO(d2d.GetDrawingText()))
        full_image.paste(sub,box=offset)
    return full_image

def mol_with_atom_index(mol):
    mol_copy = copy.deepcopy(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return mol_copy

def randomize_smiles(mol, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True)
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError("Type '{}' is not valid".format(random_type))

def get_canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def prep_mol(mol):
    Chem.RemoveStereochemistry(mol)
    mol = break_disulfide_bonds(mol)
    mol = standardize_mol(mol)
    return mol

def standardize_mol(mol):
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    # Chem.Kekulize(mol)
    # return mol

def remove_hydroxyl(smiles):
    patt = Chem.MolFromSmarts('C(=O)O')
    repl = Chem.MolFromSmiles('C=O')
    m = Chem.MolFromSmiles(smiles)
    rms = AllChem.ReplaceSubstructs(m,patt,repl)
    repl_smiles = Chem.MolToSmiles(rms[-1])
    if repl_smiles == smiles: 
        print("No carboxylic acid group")
        print(smiles)
        return smiles
    else:
        return repl_smiles

def encoded_to_fasta(encoded):
    amino_acids = amino_acid_codes()
    tokens = list(sf.split_selfies(encoded))
    regex = re.compile('[^a-zA-Z ]')
    aa_seq = [regex.sub('', t) for t in tokens]
    aa_seq = [amino_acids[aa] for aa in aa_seq if aa in amino_acids.keys()]
    return "".join(aa_seq)

def add_attachments(mol, atom_indices):
    atom_finder = re.compile(r"""
    (
    Cl? |             # Cl and Br are part of the organic subset
    Br? |
    [NOSPFIbcnosp*] | # as are these single-letter elements
    \[[^]]+\]         # everything else must be in []s
    )
    """, re.X)
    
    atom_indices.sort(reverse=True)
    smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
    # atoms = atom_finder.findall(smiles)
    ms = [x for x in atom_finder.finditer(smiles)]
    ixs = [(x.start(),x.end()) for x in ms]
    # print(ixs)
    for atom_index in atom_indices:
        order = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
        i = order.index(atom_index)
        # search_atom = atoms[i]
        # print(search_atom)
        # print(ixs[i][-1])
        
        smiles = smiles[:ixs[i][-1]] + "(*1)" + smiles[ixs[i][-1]:]
    return smiles

def get_amino_carboxyl_indices(aa_mol):
    mol = copy.deepcopy(aa_mol)
    patt = Chem.MolFromSmiles('C(C(=O))N')
    hit_ats = list(mol.GetSubstructMatch(patt))

    dict = {}
    for bond in patt.GetBonds():
        aids = [hit_ats[bond.GetBeginAtomIdx()], hit_ats[bond.GetEndAtomIdx()]]
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in aids]

        if "N" in atoms:
            dict["amino"] = aids[atoms.index("N")]
        if "O" in atoms:
            dict["carboxyl"] = aids[atoms.index("C")]
    return dict

def break_disulfide_bonds(mol):
    disulfide_bonds = []

    # Iterate through each bond and check for disulfide bonds (S-S)
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'S':
            disulfide_bonds.append((atom1.GetIdx(), atom2.GetIdx()))
    
    # Create a copy of the original molecule
    new_mol = Chem.RWMol(mol)
    
    for disulfide_bond in disulfide_bonds:
        # Break the bond
        atom_idx1, atom_idx2 = disulfide_bond
        new_mol.RemoveBond(atom_idx1, atom_idx2)
    
    return new_mol

def amino_acid_codes():
   return {
        'Alanine': 'A',
        'Arginine': 'R',
        'Asparagine': 'N',
        'Aspartic Acid': 'D',
        'Cysteine': 'C',
        'Glutamic Acid': 'E',
        'Glutamine': 'Q',
        'Glycine': 'G',
        'Histidine': 'H',
        'Isoleucine': 'I',
        'Leucine': 'L',
        'Lysine': 'K',
        'Methionine': 'M',
        'Phenylalanine': 'F',
        'Proline': 'P',
        'Serine': 'S',
        'Threonine': 'T',
        'Tryptophan': 'W',
        'Tyrosine': 'Y',
        'Valine': 'V'
        }

def amino_acid_mols(stereochem=False):
    aa = amino_acid_codes()
    aa_mols = {key: Chem.rdmolfiles.MolFromFASTA(aa[key]) for key in aa}

    if not stereochem:
        for key in aa_mols:
            Chem.RemoveStereochemistry(aa_mols[key])

    aa_mols = {key: standardize_mol(aa_mols[key]) for key in aa_mols}
    return aa_mols


def convert_smi_to_group(smi, grammar):
    try: 
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(mol)
        # mol = break_disulfide_bonds(mol)
        mol = standardize_mol(mol)

        Chem.Kekulize(mol)
        smi = Chem.MolToSmiles(mol, kekuleSmiles=True)

        prot_gr = grammar.full_encoder(mol)
        protein_mol_decoded = grammar.decoder(prot_gr)

        Chem.Kekulize(protein_mol_decoded)
        smi_gr = Chem.MolToSmiles(protein_mol_decoded, kekuleSmiles=True)

        # protein_mol_decoded = standardize_mol(protein_mol_decoded)
        # smi_gr = Chem.MolToSmiles(protein_mol_decoded)
        assert smi == smi_gr
    except:
        print("failed")
        return None
    

def convert_group_to_smi(sf_group, grammar):
    try: 
        protein_mol_decoded = grammar.decoder(sf_group)

        Chem.Kekulize(protein_mol_decoded)
        smi_gr = Chem.MolToSmiles(protein_mol_decoded, kekuleSmiles=True)

        # protein_mol_decoded = standardize_mol(protein_mol_decoded)
        # smi_gr = Chem.MolToSmiles(protein_mol_decoded)
    except:
        print("failed")
        return None
    return smi_gr


def get_mol_from_smi(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        print("conversion failed")

    if m is None:
        # print("sanitized conversion failed")
        try:
            m = Chem.MolFromSmiles(smiles, sanitize = False)
        except:
            print("conversion failed")
    
    return m


def get_num_heavy_smi(smiles):
    m = get_mol_from_smi(smiles)
    if m is not None:
        return m.GetNumHeavyAtoms()


def get_mw_smi(smiles):
    m = get_mol_from_smi(smiles)
    if m is not None:
        return Descriptors.MolWt(m)


def amino_acid_smiles(stereochem=False):

    aa = amino_acid_mols(stereochem)
    return {key: Chem.MolToSmiles(aa[key]) for key in aa}

    # return {
    #     'Alanine': 'C[C@@H](C(=O)O)N',
    #     'Arginine': 'N[C@@H](CCCNC(N)=N)C(=O)O',
    #     'Asparagine': 'C([C@@H](C(=O)O)N)C(=O)N',
    #     'Aspartic Acid': 'C([C@@H](C(=O)O)N)C(=O)O',
    #     'Cysteine': 'SC[C@H](N)C(=O)O',
    #     'Glutamine': 'N[C@@H](CCC(N)=O)C(=O)O',
    #     'Glutamic Acid': 'C(CC(=O)O)[C@@H](C(=O)O)N',
    #     'Glycine': 'C(C(=O)O)N',
    #     'Histidine': 'N[C@@H](Cc1cnc[nH]1)C(=O)O',
    #     'Isoleucine': 'CC[C@H](C)[C@@H](C(=O)O)N',
    #     'Leucine': 'CC[C@H](C)[C@@H](C(=O)O)N',
    #     'Lysine': 'N[C@@H](CCCCN)C(=O)O',
    #     'Methionine': 'CSCCC[C@H](N)C(=O)O',
    #     'Phenylalanine': 'C1=CC=C(C=C1)C[C@@H](C(=O)O)N',
    #     'Proline': 'OC(=O)[C@H]1CCCN1',
    #     'Serine': 'OC[C@H](N)C(=O)O',
    #     'Threonine': 'OCC[C@H](N)C(=O)O',
    #     'Tryptophan': 'c1ccc2c(c1)c(c[nH]2)CCC[C@H](C(=O)O)N',
    #     'Tyrosine': 'c1ccc(cc1)C[C@H](C(=O)O)N',
    #     'Valine': 'CC(C)C[C@H](C(=O)O)N'
    # }

def create_FASTA_grammar():
    constraints = group_selfies.get_preset_constraints("default")
    constraints.update({"S": 2, "S-1": 1, "S+1": 3,})
    group_selfies.set_semantic_constraints(constraints)

    # TODO: add amino acids with charged atoms
    amino_acids = amino_acid_smiles()
    amino_acids = {key: remove_hydroxyl(amino_acids[key]) for key in amino_acids}

    grammar = GroupGrammar([Group(key, amino_acids[key], all_attachment=True) for key in amino_acids])
    return grammar