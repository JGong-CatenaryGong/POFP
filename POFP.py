from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw

# patts = [
#     '[c][N]([C])[C]', #dimethylamino diethylamino
#     '[c][N]([c])[C]', #DPA
#     '[c][N]([c])[c]', #TPA
#     '[c][O][C]', #methoxy
#     '[c][O][c]', #DPO
#     '[c][S][c][c][N][c]', #phenothiazine
#     '[s]', #thiophene
#     '[o]', #furan
#     '[c][C]=[C][c]', #c-C=C-c
#     '[C]#[N]', #cyano
#     '[n+]', #aromatic positive nitrogen
#     '[o+]', #aromatic positive oxgen
#     '[B]', #Boron as acceptor
#     '[C][c](s)[n]', #benzothiazole
#     '[C][c](o)[n]', #benzoxazole
#     '[C][c](n)[n]', #benzoimidazole
#     '[n][s][n]', #aromatic N-S-N
#     '[n][se][n]', #aromatic N-Se-N
#     '[C]=[C]([N])[C](=[O])[O]', #TPA-BMO
#     '[c][C](=[O])[N][C](=[O])[c]', #bisimide
#     '[n]', #aromatic nitrogen
#     '[C]=[C]([C](=[O])[c])[C](=[O])[c]', #indanone
#     '[C]=[O]', #carboxy
#     '[c][C]#[C]', #c-C#C
#     '[c;R][n;R][c;R][c;R][n;R][c;R]', #pyrazine
#     '[C][C][C]', #non-conjugated long chain
#     # '[N]=[C;H][c][c][O;H]', #ESIPT: enol-imide
#     # '[N]=[C;H][c][c][O;H]', #ESIPT: enol-keto
#     # '[c][C](=[O])[C](=[C])[O;H]', #ESIPT: 5m ring
# ]

donor_patts = [
    '[c][N]([C])[C]', #dimethylamino diethylamino
    '[c][N]([c])[C]', #DPA
    '[c][N]([c])[c]', #TPA
    '[c][O][C]', #methoxy
    '[c][O][c]', #DPO
    '[c][S][c][c][N][c]', #phenothiazine
    '[c][O][c][c][N][c]', #phenoxazine
    '[#16]-1-[#6]=[#6]-[#6]=[#6]-1', #thiophene
    '[#8]-1-[#6]=[#6]-[#6]=[#6]-1', #furan
    '[C]=[C]', #C=C
    '[O][c][c][O]', #1,2-dihydroxy aromatic ring
    # '[c]-[c]', #biphenyl
    '[#6]-1=[#6]-[#6]-2=[#6](-[#6]=[#6]-1)-[#6]=[#6]-[#6]=[#6]-2', #naphthalene
    '[N]=[C;H][c][c][O;H]', #ESIPT: enol-imide
    '[O]=[C;H][c][c][O;H]', #ESIPT: enol-keto
    '[c][C](=[O])[C](=[C])[O;H]', #ESIPT: 5m ring
] #15

acceptor_patts = [
    '[C]#[N]', #cyano
    '[C]=[N]', #C=N
    '[n+]', #aromatic positive nitrogen
    '[o+]', #aromatic positive oxgen
    '[c][B]', #Boron as acceptor
    '[O,N][B]([F])[F]', #BODIPY substructure
    '[C][c](s)[n]', #benzothiazole
    '[C][c](o)[n]', #benzoxazole
    '[C][c](n)[n]', #benzoimidazole
    '[n][s][n]', #aromatic N-S-N
    '[c][N]=[S]=[N][c]', #NBD second form
    '[n][se][n]', #aromatic N-Se-N
    '[C]=[C]([N])[C](=[O])[O]', #TPA-BMO
    '[c][C](=[O])[N][C](=[O])[c]', #bisimide
    '[n]', #aromatic nitrogen
    '[C]=[C]([C](=[O])[c])[C](=[O])[c]', #indanone
    '[C]=[O]', #carboxy
    '[c][C]#[C]', #c-C#C
    '[N]#[C][C]([C]#[N])=[C]([c])[C](=[C])[C](c)=[C]([C]#[N])[C]#[N]', #4cyano indanone
    '[c][c][n][c][c]([c])[n,c][s][n,c][c]([c])[c][n][c][c]', #pyrazine-NBD
    '[C,c][F]', #fluoride
    '[O]=[C,c][c][c]([c])[S,s,O,o]', #flavone
    '[Au,Bi,Ir,Ru,Eu]', #metals
] #23

conjugation_breaker = [
    '[C][c][c]([c,C])[c]', #spatial hindrance between benzene rings
    '[c][c]([c])[c]([c])[c]([c])[c]([c])[c]', #BINAP
    '[c][C]([c])-[C]', #saturated aromatic bridge
    '[F][B-]([F])([F])[F]', #BF4
    '[F][P-]([F])([F])([F])([F])[F]', #PF6
    '[C]([C])([C])[C]', #t-Butyl
    '[c]([c])[c]([c])[c]([c])[c]([c])[c]([c])[c]([c])[c]([c])[c]([c])', #continual benzene rings
    # '[c]c1[c]c[c]cccc1', #multiaromatic benzene
] #7

patts = donor_patts + acceptor_patts + conjugation_breaker

aromatic_scale_index = 1

def GenSpecFP(mol, mode, with_aromaticity=True):
    mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return np.nan
    SpecFP = []
    if mode==1:
        for i in range(len(patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(patts[i])):
                SpecFP.append(1)
            else:
                SpecFP.append(0)
    elif mode==2:
        for i in range(len(patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(patts[i])):
                hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts(patts[i]))
                dimen = np.array(hit_ats).shape
                num = dimen[0]
                SpecFP.append(num)
            else:
                SpecFP.append(0)
    if with_aromaticity:
        if mode==1:
            if mol.HasSubstructMatch(Chem.MolFromSmarts('a')):
                SpecFP.append(1)
            else:
                SpecFP.append(0)
        elif mode==2:
            if mol.HasSubstructMatch(Chem.MolFromSmarts('a')):
                hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts('a'))
                dimen = np.array(hit_ats).shape
                num = dimen[0]/aromatic_scale_index
                SpecFP.append(num)
            else:
                SpecFP.append(0)
    return np.array(SpecFP)

def GenSpecMat(mol, mode, normalize=False, with_aromaticity=True):
    mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return np.nan
    donorfp = []
    if mode==1:
        for i in range(len(donor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(donor_patts[i])):
                donorfp.append(1)
            else:
                donorfp.append(0)
    elif mode==2:
        for i in range(len(donor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(donor_patts[i])):
                hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts(donor_patts[i]))
                dimen = np.array(hit_ats).shape
                num = dimen[0]
                donorfp.append(num)
            else:
                donorfp.append(0)
    if with_aromaticity:
        if mode==1:
            if mol.HasSubstructMatch(Chem.MolFromSmarts('a')):
                donorfp.append(1)
            else:
                donorfp.append(0)
        elif mode==2:
            if mol.HasSubstructMatch(Chem.MolFromSmarts('a')):
                hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts('a'))
                dimen = np.array(hit_ats).shape
                num = dimen[0]/aromatic_scale_index
                # print(num)
                donorfp.append(num)
            else:
                donorfp.append(0)
    acceptorfp = []
    if mode==1:
        for i in range(len(acceptor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(acceptor_patts[i])):
                acceptorfp.append(1)
            else:
                acceptorfp.append(0)
    elif mode==2:
        for i in range(len(acceptor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(acceptor_patts[i])):
                hit_ats2 = mol.GetSubstructMatches(Chem.MolFromSmarts(acceptor_patts[i]))
                dimen2 = np.array(hit_ats2).shape
                num2 = dimen2[0]
                acceptorfp.append(num2)
            else:
                acceptorfp.append(0)
    mat = np.zeros((len(donorfp),len(acceptorfp)))
    for i in range(len(donorfp)):
        for j in range(len(acceptorfp)):
            mat[i,j] = donorfp[i]*acceptorfp[j]
    if normalize:
        max = mat.max()
        min = mat.min()
        if max != 0:
            mat = mat/(max-min) + min
        else:
            pass
    else:
        pass
    return mat

def GenSpecPlusMat(mol, mode, normalize=False, with_aromaticity=True):
    if mol is None:
        return np.nan
    mol = Chem.MolFromSmiles(mol)
    donorfp = []
    if mode==1:
        for i in range(len(donor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(donor_patts[i])):
                donorfp.append(1)
            else:
                donorfp.append(0)
    elif mode==2:
        for i in range(len(donor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(donor_patts[i])):
                hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts(donor_patts[i]))
                dimen = np.array(hit_ats).shape
                num = dimen[0]
                donorfp.append(num)
            else:
                donorfp.append(0)
    if with_aromaticity:
        if mode==1:
            if mol.HasSubstructMatch(Chem.MolFromSmarts('a')):
                donorfp.append(1)
            else:
                donorfp.append(0)
        elif mode==2:
            if mol.HasSubstructMatch(Chem.MolFromSmarts('a')):
                hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts('a'))
                dimen = np.array(hit_ats).shape
                num = dimen[0]/aromatic_scale_index
                # print(num)
                donorfp.append(num)
            else:
                donorfp.append(0)
    acceptorfp = []
    if mode==1:
        for i in range(len(acceptor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(acceptor_patts[i])):
                acceptorfp.append(1)
            else:
                acceptorfp.append(0)
    elif mode==2:
        for i in range(len(acceptor_patts)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(acceptor_patts[i])):
                hit_ats2 = mol.GetSubstructMatches(Chem.MolFromSmarts(acceptor_patts[i]))
                dimen2 = np.array(hit_ats2).shape
                num2 = dimen2[0]
                acceptorfp.append(num2)
            else:
                acceptorfp.append(0)
    mat = np.zeros((len(donorfp),len(acceptorfp)))
    for i in range(len(donorfp)):
        for j in range(len(acceptorfp)):
            mat[i,j] = donorfp[i]+acceptorfp[j]
    if normalize:
        max = mat.max()
        min = mat.min()
        if max != 0:
            mat = mat/(max-min) + min
        else:
            pass
    else:
        pass
    return mat

def GenDA(mol):
    mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return np.nan
    donors = 0
    acceptors = 0
    for i in range(len(donor_patts)):
        donor = hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts(donor_patts[i]))
        dimen = np.array(hit_ats).shape
        num = dimen[0]
        donors += num
    for j in range(len(acceptor_patts)):
        acceptor = hit_ats = mol.GetSubstructMatches(Chem.MolFromSmarts(acceptor_patts[j]))
        dimen2 = np.array(hit_ats).shape
        num2 = dimen2[0]
        acceptors += num2

    return donors, acceptors



# some_chemicals =[
#                 'CCC[n+]6cc4ccc(/C(=C(c1ccc(OC)cc1)\c2ccc(OC)cc2)c3ccccc3)cc4c(c5ccccc5)c6c7ccccc7',
#                 'c1c2N=S=Nc2cc2N=S=Nc12',
#                 'c4coc(/C(=C(c1ccco1)/c2ccco2)c3ccco3)c4',
#                 'c1(/C(c2ccccc2)=C(c3ccccc3)/c4ccccc4)ccccc1'
#                 ]

# some_chemicals = list(map(Chem.MolFromSmiles, some_chemicals))
# print(some_chemicals)

# print(GenSpecFP(some_chemicals[3], mode=2))