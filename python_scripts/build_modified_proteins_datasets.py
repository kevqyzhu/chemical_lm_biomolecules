import os, copy, random, itertools, rdkit, traceback, multiprocessing

import pandas as pd

from rdkit import Chem

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

M = Chem.MolFromSmarts('[NX3;$([N][CX3](=[OX1])[C][NX3,NX4+])][C][CX3](=[OX1])[NX3;$([N][C][CX3](=[OX1])[NX3,OX2,OX1-])]') # finds peptidic structures which are neither C- nor N-terminal. Both neighbours must be amino-acids/peptides
C = Chem.MolFromSmarts('[NX3;$([N][CX3](=[OX1])[C][NX3,NX4+])][C][CX3](=[OX1])[OX2H,OX1-]') # finds C-terminal amino acids
N = Chem.MolFromSmarts('[NX3,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])][C][CX3](=[OX1])[NX3;$([N][C][CX3](=[OX1])[NX3,OX2,OX1-])]') # finds N-terminal amino acids. As above, N may be substituted, but not part of an amide-bond.
D = Chem.MolFromSmarts('[#16X2H0][#16X2H0]')

sidechain_smarts = {
'W':'[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12',
'Y':'[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1',
'K':'[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]',
'E':'[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]',
'Q':'[CH2X4][CH2X4][CX3](=[OX1])[NX4+,NX3+0]',
'M':'[CH2X4][CH2X4][SX2][CH3X4]',
'L':'[CH2X4][CHX4]([CH3X4])[CH3X4]',
'I':'[CHX4]([CH3X4])[CH2X4][CH3X4]',
'D':'[CH2X4][CX3](=[OX1])[OH0-,OH]',
'V':'[CHX4]([CH3X4])[CH3X4]',
'T':'[CHX4]([CH3X4])[OX2H]',
'P':'N1[CX4H]([CH2][CH2][CH2]1)',
'F':'[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1', # '[CH2X4][cX3]1[cX3H][cX3H][cX3][cX3H][cX3H]1'
'H':'[#6]-[#6]1:[#6]:[#7]:[#6]:[#7H]:1', # '[#6]-[#6]1:[#6]:[#7H]:[#6]:[#7]:1'
'N':'[CH2X4][CX3](=[OX1])[NX4+,NX3+0]', #FULL '[#8]=[#6](-[#7])-[#6]-[#6@H](-[#7])-[#6](=[#8])-[#8]', # '[CH2X4][CX3](=[OX1])[OH0-,OH]', [NX4+,NX3+0]
'N.':'[CH2X4][CX3]([NX4+,NX3+0])=[OX1]',
'R':'[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]',
'R.':'[CH2X4][CH2X4][CH2X4][NX3+,NX2+0]=[CH0X3]([NH2X3])[NH2X3]',
'R..':'[CH2X4][CH2X4][CH2X4][NHX3][CH0X3]([NH2X3])=[NH2X3+,NHX2+0]',
'R...':'[#6]-[#6]-[#6]-[#7]-[#6](-[#7])-[#7]' # smiles CCCNC(N)N chain
}

sidechain_from_smarts = {a:Chem.MolFromSmarts(s) for a,s in sidechain_smarts.items()}
scs = list(sidechain_from_smarts.keys())

sidechain_smiles = {
'[W]':['CC1=c2ccccc2=NC1','CC1=CN=C2CC=CC=C12','CC1=CN=C2C=CC=CC12'],
'[Y]':['Cc1ccc(O)cc1'],
'[F]':['Cc1ccccc1'],
'[K]':['CCCCN'],
'[E]':['CCC(=O)O', 'C=CC(=O)O'],
'[Q]':['CCC(=O)N', 'CCC(N)=O'],
'[M]':['CCSC','C=CSC'],
'[L]':['CC(C)C'],
'[I]':['C(C)CC','CCCC','C=CCC'],
'[D]':['CC(=O)O'],
'[V]':['C(C)C','CCC'],
'[T]':['CCO'],
'[H]':['Cc1c[nH]cn1','CC1=NC=NC1','Cc1cnc[nH]1','CC1=CNCN1','CC1C=NC=N1'],
'[N]':['CC(N)=O'],
'[R]':['CCCNC(N)N','CCCN=C(N)N'],
'[C]':['CS'],
'[S]':['CO']
}

restype_1to3 = {'[A]': '[ALA]', '[R]': '[ARG]', '[N]': '[ASN]', '[D]': '[ASP]',
                '[C]': '[CYS]', '[Q]': '[GLN]', '[E]': '[GLU]', '[G]': '[GLY]',
                '[H]': '[HIS]', '[I]': '[ILE]', '[L]': '[LEU]', '[K]': '[LYS]',
                '[M]': '[MET]', '[F]': '[PHE]', '[P]': '[PRO]', '[S]': '[SER]',
                '[T]': '[THR]', '[W]': '[TRP]', '[Y]': '[TYR]', '[V]': '[VAL]'}

#each backbone group of amino-carboxyl has atom idx --> (N, C (sidechain), C=O, N)

def get_sidechain_atoms(protein_mol, atoms_ac):
    alpha_carbon = atoms_ac[1]
    return list(get_neighbors(protein_mol, alpha_carbon, visited=set(atoms_ac)))

def get_first_order_neighbors(atom, exclude):
    return [atom.GetIdx() for atom in atom.GetNeighbors() if atom.GetIdx() not in exclude]

def get_formula_from_atoms(mol, sidechain_atoms):
    elements = [mol.GetAtomWithIdx(atom_idx).GetSymbol() for atom_idx in sidechain_atoms]
    return (elements.count('C'), elements.count('N'),
            elements.count('O'), elements.count('S'))



def remove_sidechain_sidechain_connects(mol):
    cys2cys_connects = mol.GetSubstructMatches(D)
    print(cys2cys_connects)
    for cys2cys in cys2cys_connects:
        emol = Chem.EditableMol(mol)
        emol.RemoveBond(*cys2cys)
        mol = emol.GetMol()
    return mol

################################################################################

def get_neighbors(mol, atom_idx, visited=set()):
    """Recursively find all neighbors of neighbors starting from an atom index."""
    visited.add(atom_idx)
    neighbors = set([atom_idx])
    #print(mol,visited,neighbors)
    for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx not in visited:
            sub_neighbors = get_neighbors(mol, neighbor_idx, visited)
            neighbors.update(sub_neighbors)
    return neighbors

def get_sidechain(protein_mol, atoms_ac):
    #atoms_ac is 5 amide carboxyl group atoms
    return list(get_neighbors(protein_mol, atoms_ac[1], visited=set(atoms_ac)))

def all_atoms_common(atoms, sidechain_atoms): # aas is list of atoms
    common = set(atoms).intersection(set(sidechain_atoms))
    return len(common)==len(sidechain_atoms) or len(atoms)==len(sidechain_atoms)

def sidechain_match(sidechain_atoms, matched_atoms):
    #print(matched_atoms, sidechain_atoms)
    if any([all_atoms_common(atoms, sidechain_atoms) for atoms in matched_atoms]):
        return True
    return False

################################################################################

def get_protein_attachment_idx(protein_mol, atom_idx_tuple, exclude=None):
    atoms = [protein_mol.GetAtomWithIdx(int(atom_idx)) for atom_idx in atom_idx_tuple]
    if len(atoms)==1:
        return atoms[0].GetIdx()
    atoms = [atom for atom in atoms if atom.GetTotalNumHs()>0]
    atoms = [atom for atom in atoms if atom.GetIdx() not in exclude]
    #atoms = [atom for atom in atoms if atom.GetSymbol()=='C']
    return random.choice(atoms).GetIdx()

def random_sidechain_atomid(sc_mod_mol):
    atoms = [atom for atom in sc_mod_mol.GetAtoms() if atom.GetTotalNumHs()>0]
    return random.choice(atoms).GetIdx()

def modify_protein_sidechain(protein_mol, backbone_atoms,
                             attachment, sidechain_modification_mol):
    #print('schain',attachment)
    protein_attach_idx = get_protein_attachment_idx(protein_mol, attachment, backbone_atoms)
    sc_attach_protein = random_sidechain_atomid(sidechain_modification_mol)
    new_mol = Chem.CombineMols(protein_mol, sidechain_modification_mol)
    sc_attach = sc_attach_protein+protein_mol.GetNumAtoms()
    emol= Chem.RWMol(new_mol)
    emol.AddBond(protein_attach_idx, sc_attach, order=Chem.rdchem.BondType.SINGLE)
    return emol.GetMol()

modifications_file = 'zinc.txt'
#modifications_file = 'sidechain_modifications.txt'

sidechain_modifications = pd.read_csv(modifications_file)['smiles']
if 'zinc' not in modifications_file:
    sidechain_modifications = [Chem.MolFromSmiles(s) for s in sidechain_modifications]

def random_protein_modification(protein_smi):

    protein_mol = Chem.MolFromSmiles(protein_smi)
    protein_mol = remove_sidechain_sidechain_connects(protein_mol)

    npa = protein_mol.GetNumAtoms()
    backbone = list(protein_mol.GetSubstructMatches(M))#; print('backbone length',len(backbone))
    sidechain_atoms = [get_sidechain(protein_mol, atoms) for atoms in backbone]

    backbone_atoms = [atom_idx for atoms in backbone for atom_idx in atoms]

    #print('sc atoms', sidechain_atoms)
    if 'zinc' not in modifications_file:
        sc_mods = [random.choice(sidechain_modifications) for _ in range(len(backbone))]
        for mod, sidechain in zip(sc_mods, sidechain_atoms):
            protein_mol = modify_protein_sidechain(protein_mol, backbone_atoms, sidechain, mod)
    else:
        #print("attaching to a single sidechain")
        attachable_residue='K' # change which residue to attach to
        raw_sc_mol = sidechain_from_smarts[attachable_residue]
        matches = list(protein_mol.GetSubstructMatches(raw_sc_mol))
        def sc_match(sc): return sidechain_match(sc, matches)
        sidechain_atoms = list(filter(sc_match, sidechain_atoms))#; print("sidechains found:",sidechain_atoms)
        sidechain = random.choice(sidechain_atoms)
        mod = Chem.MolFromSmiles(random.choice(sidechain_modifications))
        protein_mol = modify_protein_sidechain(protein_mol, backbone_atoms, sidechain, mod)

    nda = protein_mol.GetNumAtoms()
    facts = "backbone {} | {} atoms in original protein : --> {} atoms in modified protein".format(len(backbone), npa,nda)
    print(facts)
    #print('new bb length',len(protein_mol.GetSubstructMatches(M)))

    smi = Chem.MolToSmiles(protein_mol)
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    #print('smiles : ',smi)

    return smi

def log(fname, text):
    with open(fname, 'a') as f:
        f.write(text + '\n')

file = 'protein_drug_conjugates.txt' if 'zinc' in modifications_file else  'unnatural_proteins.txt'

def mod_protein(smile):
    try: # set nmods =0 if you want to attach to all sidechains
        smi = random_protein_modification(smile)
        log(file, smi)
        return smi
    except:
        return ""

def test():
    file = 'single_chains_0_250_AA_95_sim_rdkit_0_2500_DS.csv'
    file = 'single_chains.csv' # 20 MB file with 0-250 residue chains
    import pandas as pd
    from primary_sequence import determine_primary_structure
    df = pd.read_csv(file)

    #print(df)
    n = random.choice(range(len(df)))
    ids, smiles = df['pdb_id'], df['smiles']
    id, smi = ids[n], smiles[n]
    print(id)
    #print(smi)
    print('protein sequence')
    print(determine_primary_structure(smi))
    print('modifying protein')
    print(random_protein_modification(smi, sidechain_modifications,
                                      nmods = 0, attachable_residue='K'))

if __name__ == "__main__":
    #test(); exit()

    train_file = 'single_chains_0_250_AA_95_sim_rdkit_0_2500_DS.csv'
    train_file = 'single_chains.csv' # 20 MB file with 0-250 residue chains

    protein_smiles = pd.read_csv(train_file)['smiles'].tolist()
    protein_smiles = random.sample(protein_smiles, 100)

    with multiprocessing.Pool() as pool:
        pool.map(mod_protein, protein_smiles)
