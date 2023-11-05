import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem ,AddHs
from rdkit.Chem import Descriptors, Lipinski , Descriptors3D
from rdkit.Chem import rdMolDescriptors, GraphDescriptors
from rdkit.Chem import MolFromSmiles ,MolFromSmarts

from rdkit.Chem import rdMolDescriptors, GraphDescriptors, Fragments
# from self_functions import *
import warnings
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def rd_MolDescriptors(smiles):
    columns = [
        "SMILES",
        "LogP",
        "Molar_Refractivity",
        "ExactMolecular_Weight",
        "FractionC_inSP3-hyb",
        "HallKierAlpha",
        "LabuteASA",
        "#Aliphatic_carbo_cycles",
        "#Aliphatic_hetro_cycles",
        "#Aliphatic_rings",
        "#Amide_bonds",
        "#Aromatic_carbo_cycles",
        "#Aromatic_hetro_cycles",
        "#Aromatic_rings",
        "#Atoms",
        "#BridgeHead_atoms",
        "#H-Bond_Acceptors",
        "#H-Bond_donors",
        "#HeavyAtoms",
        "#HetroAtoms",
        "#Hetrocycles",
        "Lipinski-H-Bond_Acceptors",
        "Lipinski-H-Bond_donors",
        "#Rings",
        "#RotatableBonds",
        "Saturated_carbo_cycles",
        "Saturated_hetro_cycles",
        "Saturated_rings",
        "Spinors",
        "Phi",
        "The_polar_surface_area",
    ]
    re_moldiesc = pd.DataFrame(columns=columns)
    for i in smiles:
        mol_i = Chem.AddHs(Chem.MolFromSmiles(i))
        AllChem.EmbedMolecule(mol_i)
        re_moldiesc.loc[len(re_moldiesc.index), columns] = [
            i,
            rdMolDescriptors.CalcCrippenDescriptors(mol_i)[0],
            rdMolDescriptors.CalcCrippenDescriptors(mol_i)[1],
            rdMolDescriptors.CalcExactMolWt(mol_i),
            rdMolDescriptors.CalcFractionCSP3(mol_i),
            rdMolDescriptors.CalcHallKierAlpha(mol_i),
            rdMolDescriptors.CalcLabuteASA(mol_i),
            rdMolDescriptors.CalcNumAliphaticCarbocycles(mol_i),
            rdMolDescriptors.CalcNumAliphaticHeterocycles(mol_i),
            rdMolDescriptors.CalcNumAliphaticRings(mol_i),
            rdMolDescriptors.CalcNumAmideBonds(mol_i),
            rdMolDescriptors.CalcNumAromaticCarbocycles(mol_i),
            rdMolDescriptors.CalcNumAromaticHeterocycles(mol_i),
            rdMolDescriptors.CalcNumAromaticRings(mol_i),
            rdMolDescriptors.CalcNumAtoms(mol_i),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol_i),
            rdMolDescriptors.CalcNumHBA(mol_i),
            rdMolDescriptors.CalcNumHBD(mol_i),
            rdMolDescriptors.CalcNumHeavyAtoms(mol_i),
            rdMolDescriptors.CalcNumHeteroatoms(mol_i),
            rdMolDescriptors.CalcNumHeterocycles(mol_i),
            rdMolDescriptors.CalcNumLipinskiHBA(mol_i),
            rdMolDescriptors.CalcNumLipinskiHBD(mol_i),
            rdMolDescriptors.CalcNumRings(mol_i),
            rdMolDescriptors.CalcNumRotatableBonds(mol_i),
            rdMolDescriptors.CalcNumSaturatedCarbocycles(mol_i),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol_i),
            rdMolDescriptors.CalcNumSaturatedRings(mol_i),
            rdMolDescriptors.CalcNumSpiroAtoms(mol_i),
            rdMolDescriptors.CalcPhi(mol_i),
            rdMolDescriptors.CalcTPSA(mol_i),
        ]
    return re_moldiesc

def descriptors(smiles):
    columns = [
        "Exact_Molecular_Weight",
        "HeavyAtom_Molecular_Weight",
        "FpDensityMorgan1",
        "FpDensityMorgan2",
        "FpDensityM3",
        "MaxAbsPartialCharge",
        "MaxPartialCharge",
        "MinAbsPartialCharge",
        "MinPartialCharge",
        "RadicalElectrons",
        "ValenceElectrons",
        "Avg_Molecular_Weight",
    ]
    re_moldiesc = pd.DataFrame(columns=columns)
    for i in smiles:
        mol_i = Chem.AddHs(Chem.MolFromSmiles(i))
        AllChem.EmbedMolecule(mol_i)
        re_moldiesc.loc[len(re_moldiesc.index), columns] = [
            Descriptors.ExactMolWt(mol_i),
            Descriptors.HeavyAtomMolWt(mol_i),
            Descriptors.FpDensityMorgan1(mol_i),
            Descriptors.FpDensityMorgan2(mol_i),
            Descriptors.FpDensityMorgan3(mol_i),
            Descriptors.MaxAbsPartialCharge(mol_i),
            Descriptors.MaxPartialCharge(mol_i),
            Descriptors.MinAbsPartialCharge(mol_i),
            Descriptors.MinPartialCharge(mol_i),
            Descriptors.NumRadicalElectrons(mol_i),
            Descriptors.NumValenceElectrons(mol_i),
            Descriptors.MolWt(mol_i),
        ]
    return re_moldiesc

def lipinski(smiles):
    columns = [
        "Fraction_C_SP3-hyb",
        "HeavyAtomCount",
        "NHOHCount",
        "NOCount",
        "Aliphatic_carbo_cycles",
        "Aliphatic_hetro_cycles",
        "AliphaticRings",
        "Aromatic_carbo_cycles",
        "Aromatic_hetro_cycles",
        "#AromaticRings",
        "#HAcceptors",
        "#HDonors",
        "#Heteroatoms",
        "#RotatableBonds",
        "RingCount",
    ]
    re_moldiesc = pd.DataFrame(columns=columns)
    for i in smiles:
        mol_i = Chem.AddHs(Chem.MolFromSmiles(i))
        AllChem.EmbedMolecule(mol_i)
        re_moldiesc.loc[len(re_moldiesc.index), columns] = [
            Lipinski.FractionCSP3(mol_i),
            Lipinski.HeavyAtomCount(mol_i),
            Lipinski.NHOHCount(mol_i),
            Lipinski.NOCount(mol_i),
            Lipinski.NumAliphaticCarbocycles(mol_i),
            Lipinski.NumAliphaticHeterocycles(mol_i),
            Lipinski.NumAliphaticRings(mol_i),
            Lipinski.NumAromaticCarbocycles(mol_i),
            Lipinski.NumAromaticHeterocycles(mol_i),
            Lipinski.NumAromaticRings(mol_i),
            Lipinski.NumHAcceptors(mol_i),
            Lipinski.NumHDonors(mol_i),
            Lipinski.NumHeteroatoms(mol_i),
            Lipinski.NumRotatableBonds(mol_i),
            Lipinski.RingCount(mol_i),
        ]
    return re_moldiesc

def graph_descp(smiles):
    columns = ["BalabanJ", "BertzCT",'chi0']
    re_moldiesc = pd.DataFrame(columns=columns)
    for i in smiles:
        mol_i = Chem.AddHs(Chem.MolFromSmiles(i))
        AllChem.EmbedMolecule(mol_i)
        re_moldiesc.loc[len(re_moldiesc.index), columns] = [
            GraphDescriptors.BalabanJ(mol_i),
            GraphDescriptors.BertzCT(mol_i),
            GraphDescriptors.Chi0(mol_i),
        ]
    return re_moldiesc

def mole_property_calc(smiles):
    molecular_property_dict = {#"Exact Molecular Weight": rdMolDescriptors.CalcExactMolWt,
                           #"Heavy Atom Molecular Weight": Descriptors.HeavyAtomMolWt,
                           #"Average Molecular Weight": Descriptors.MolWt,
                           "LogP": Chem.Crippen.MolLogP,
                           "Molar Refracticity": Chem.Crippen.MolMR,
                           #"Labute ASA": rdMolDescriptors.CalcLabuteASA,
                           #"Number of Rings": rdMolDescriptors.CalcNumRings,
                          #  "Number of Hetro-cycles": rdMolDescriptors.CalcNumHeterocycles,
                          #  "Number of Aliphatic Carbo-cycles": rdMolDescriptors.CalcNumAliphaticCarbocycles,
                          #  "Number of Aliphatic hetro cycles": rdMolDescriptors.CalcNumAliphaticHeterocycles,
                          #  "Number of Aliphatic rings": rdMolDescriptors.CalcNumAliphaticRings,
                          #  "Number of Aromatic carbo cycles": rdMolDescriptors.CalcNumAromaticCarbocycles,
                          #  "Number of Aromatic hetro cycles": rdMolDescriptors.CalcNumAromaticHeterocycles,
                          #  "Number of Aromatic rings": rdMolDescriptors.CalcNumAromaticRings,
                          #  "Number of saturated carbo cycles": rdMolDescriptors.CalcNumSaturatedCarbocycles,
                          #  "Number of saturated hetro cycles": rdMolDescriptors.CalcNumSaturatedHeterocycles,
                          #  "Number of saturated rings": rdMolDescriptors.CalcNumSaturatedRings,
                          #  "Fb Density Morgan 1": Descriptors.FpDensityMorgan1,
                          #  "Fb Density Morgan 2": Descriptors.FpDensityMorgan2,
                          #  "Fb Density Morgan 3": Descriptors.FpDensityMorgan3,
                          #  "Maximum absolute partial charge": Descriptors.MaxAbsPartialCharge,
                          #  "Maximum partial charge": Descriptors.MaxPartialCharge,
                          #  "Minimum absolute partial charge": Descriptors.MinAbsPartialCharge,
                          #  "Minimum partial charge": Descriptors.MinPartialCharge,
                          #  "Number of Radical electrons": Descriptors.NumRadicalElectrons,
                          #  "Number of Valence electrons": Descriptors.NumValenceElectrons,
                          #  "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
                          #  "BertzCT": GraphDescriptors.BertzCT,
                          #  "BalabanJ": GraphDescriptors.BalabanJ,
                          #  "TPSA": rdMolDescriptors.CalcTPSA,
                          #  "HallKierAlpha": rdMolDescriptors.CalcHallKierAlpha,
                           "PBF": rdMolDescriptors.CalcPBF,
                          #  "Phi": rdMolDescriptors.CalcPhi,
                           "MaxAbsEStateIndex": Chem.EState.EState.MaxAbsEStateIndex,
                           "MaxEStateIndex": Chem.EState.EState.MaxEStateIndex,
                           "MinEStateIndex": Chem.EState.EState.MinEStateIndex,
                           "MinAbsEStateIndex": Chem.EState.EState.MinAbsEStateIndex,
                           "PMI1": rdMolDescriptors.CalcPMI1,
                           "PMI2": rdMolDescriptors.CalcPMI2,
                           "PMI3": rdMolDescriptors.CalcPMI3,
                           "NPR1": rdMolDescriptors.CalcNPR1,
                           "NPR2": rdMolDescriptors.CalcNPR2,
                           "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration,
                           "InertialShapeFactor": Descriptors3D.InertialShapeFactor,
                           "Eccentricity": Descriptors3D.Eccentricity,
                           "Asphericity": Descriptors3D.Asphericity,
                           "SpherocityIndex": Descriptors3D.SpherocityIndex,
                           "Kappa1": rdMolDescriptors.CalcKappa1,
                           "Kappa2": rdMolDescriptors.CalcKappa2,
                           "Kappa3": rdMolDescriptors.CalcKappa3
                           }
    molecular_property_dataframe=pd.DataFrame()
    for key, value in molecular_property_dict.items():
        molecular_property_dataframe["mol"] = smiles.map(MolFromSmiles).map(Chem.AddHs)
        molecular_property_dataframe["mol"].map(AllChem.EmbedMolecule)
        molecular_property_dataframe[key] = molecular_property_dataframe["mol"].map(value)
    molecular_property_dataframe.drop(["mol"], inplace=True, axis=1)
    return molecular_property_dataframe

def atomic_property_calc(smiles):
    atomic_properties_dict = {#"Number of atoms": rdMolDescriptors.CalcNumAtoms,
                          # "Number of Bridge head atoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
                          # "Number of Heavy atoms": rdMolDescriptors.CalcNumHeavyAtoms,
                          # "Number of Hetro atoms": rdMolDescriptors.CalcNumHeteroatoms,
                          # "Number of Spiro atoms": rdMolDescriptors.CalcNumSpiroAtoms,
                          "Number of unspecified atomic stereocenters": rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters,
                          # "Atom pair atom code": rdMolDescriptors.GetAtomPairAtomCode,
                          # "Atom Pair Code": rdMolDescriptors.GetAtomPairCode
                          }
    atomic_prop_dataframe=pd.DataFrame()
    for key, value in atomic_properties_dict.items():
        atomic_prop_dataframe["mol"] = smiles.map(MolFromSmiles)
        atomic_prop_dataframe["mol"].map(AllChem.EmbedMolecule)
        atomic_prop_dataframe[key] =atomic_prop_dataframe["mol"].map(value)
    atomic_prop_dataframe.drop(["mol"], inplace=True, axis=1)

    return atomic_prop_dataframe

def peroxide_count(mol):
    # mol = MolFromSmiles(smile)
    functional_group = MolFromSmarts('[OX2,OX1-][OX2,OX1-]')
    match = mol.GetSubstructMatches(functional_group)
    return len(match)

def ortho_ring_count(mol):
    # mol = MolFromSmiles(smile)
    functional_group = MolFromSmarts('[OH]-!:aa-!:[OH]')
    match = mol.GetSubstructMatches(functional_group)
    return len(match)

def meta_ring_count(mol):
    # mol = MolFromSmiles(smile)
    functional_group = MolFromSmarts('[OH]-!:aaa-!:[OH]')
    match = mol.GetSubstructMatches(functional_group)
    return len(match)

def para_ring_count(mol):
    # mol = MolFromSmiles(smile)
    functional_group = MolFromSmarts('[OH]-!:aaaa-!:[OH]')
    match = mol.GetSubstructMatches(functional_group)
    return len(match)


def functional_group_calc(smiles):
    functional_group_dict = {
                              # "Number of NHOH": Lipinski.NHOHCount,
                            #  "Number of NO": Lipinski.NOCount,
                            "Number of Aliphatic COOH": Fragments.fr_Al_COO,
                            "Number of Aliphatic OH": Fragments.fr_Al_OH,
                            "Number of Aliphatic OH (not tert)": Fragments.fr_Al_OH_noTert,
                            "Number of Aromatic Functional groups": Fragments.fr_ArN,
                            "Number of Aromatic COOH": Fragments.fr_Ar_COO,
                            "Number of Aromatic Nitrogens": Fragments.fr_Ar_N,
                            "Number of Aromatic amines": Fragments.fr_Ar_NH,
                            "Number of Aromatic OH": Fragments.fr_Ar_OH,
                            "Number of COOH": Fragments.fr_COO,
                            "Number of carbonyl O": Fragments.fr_C_O,
                            "Number of carbonyl O(Excluding COOH)": Fragments.fr_C_O_noCOO,
                            "Number of thiocarbonyl": Fragments.fr_C_S,
                            "Number of C(OH)CCN-Ctert-alkyl or C(OH)CCNcyclic": Fragments.fr_HOCCN,
                            "Number of Imines": Fragments.fr_Imine,
                            "Number of Tertiary amines": Fragments.fr_NH0,
                            "Number of Secondary amines": Fragments.fr_NH1,
                            "Number of Primary amines": Fragments.fr_NH2,
                            "Number of hydroxylamine groups": Fragments.fr_N_O,
                            "Number of XCCNR groups": Fragments.fr_Ndealkylation1,
                            "Number of tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)": Fragments.fr_Ndealkylation2,
                            "Number of H-pyrrole nitrogens": Fragments.fr_Nhpyrrole,
                            "Number of thiol groups": Fragments.fr_SH,
                            "Number of aldehydes": Fragments.fr_aldehyde,
                            "Number of alkyl carbamates (subject to hydrolysis)": Fragments.fr_alkyl_carbamate,
                            "Number of alkyl halides": Fragments.fr_alkyl_halide,
                            "Number of allylic oxidation sites excluding steroid dienone": Fragments.fr_allylic_oxid,
                            "Number of amides": Fragments.fr_amide,
                            "Number of amidine groups": Fragments.fr_amidine,
                            "Number of anilines": Fragments.fr_aniline,
                            "Number of aryl methyl sites for hydroxylation": Fragments.fr_aryl_methyl,
                            "Number of azide groups": Fragments.fr_azide,
                            "Number of azo groups": Fragments.fr_azo,
                            "Number of barbiturate groups": Fragments.fr_barbitur,
                            "Number of benzene rings": Fragments.fr_benzene,
                            "Number of benzodiazepines with no additional fused rings": Fragments.fr_benzodiazepine,
                            "Bicyclic": Fragments.fr_bicyclic,
                            "Number of diazo groups": Fragments.fr_diazo,
                            "Number of dihydropyridines": Fragments.fr_dihydropyridine,
                            "Number of epoxide rings": Fragments.fr_epoxide,
                            "Number of esters": Fragments.fr_ester,
                            "Number of ether oxygens (including phenoxy)": Fragments.fr_ether,
                            "Number of furan rings": Fragments.fr_furan,
                            "Number of guanidine groups": Fragments.fr_guanido,
                            "Number of halogens": Fragments.fr_halogen,
                            "Number of hydrazine groups": Fragments.fr_hdrzine,
                            "Number of hydrazone groups": Fragments.fr_hdrzone,
                            "Number of imidazole rings": Fragments.fr_imidazole,
                            "Number of imide groups": Fragments.fr_imide,
                            "Number of isocyanates": Fragments.fr_isocyan,
                            "Number of isothiocyanates": Fragments.fr_isothiocyan,
                            "Number of ketones": Fragments.fr_ketone,
                            "Number of ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha": Fragments.fr_ketone_Topliss,
                            "Number of beta lactams": Fragments.fr_lactam,
                            "Number of cyclic esters (lactones)": Fragments.fr_lactone,
                            "Number of methoxy groups -OCH3": Fragments.fr_methoxy,
                            "Number of morpholine rings": Fragments.fr_morpholine,
                            "Number of nitriles": Fragments.fr_nitrile,
                            "Number of nitro groups": Fragments.fr_nitro,
                            "Number of nitro benzene ring substituents": Fragments.fr_nitro_arom,
                            "Number of non-ortho nitro benzene ring substituents": Fragments.fr_nitro_arom_nonortho,
                            "Number of nitroso groups, excluding NO2": Fragments.fr_nitroso,
                            "Number of oxazole rings": Fragments.fr_oxazole,
                            "Number of oxime groups": Fragments.fr_oxime,
                            "Number of para-hydroxylation sites": Fragments.fr_para_hydroxylation,
                            "Number of phenols": Fragments.fr_phenol,
                            "Number of phenolic OH excluding ortho intramolecular Hbond substituents": Fragments.fr_phenol_noOrthoHbond,
                            "Number of phosphoric acid groups": Fragments.fr_phos_acid,
                            "Number of phosphoric ester groups": Fragments.fr_phos_ester,
                            "Number of piperdine rings": Fragments.fr_piperdine,
                            "Number of piperzine rings": Fragments.fr_piperzine,
                            "Number of primary amides": Fragments.fr_priamide,
                            "Number of primary sulfonamides": Fragments.fr_prisulfonamd,
                            "Number of pyridine rings": Fragments.fr_pyridine,
                            "Number of quaternary nitrogens": Fragments.fr_quatN,
                            "Number of thioether": Fragments.fr_sulfide,
                            "Number of sulfonamides": Fragments.fr_sulfonamd,
                            "Number of sulfone groups": Fragments.fr_sulfone,
                            "Number of terminal acetylenes": Fragments.fr_term_acetylene,
                            "Number of tetrazole rings": Fragments.fr_tetrazole,
                            "Number of thiazole rings": Fragments.fr_thiazole,
                            "Number of thiocyanates": Fragments.fr_thiocyan,
                            "Number of thiophene rings": Fragments.fr_thiophene,
                            "Number of unbranched alkanes of at least 4 members (excludes halogenated alkanes)": Fragments.fr_unbrch_alkane,
                            "Number of urea groups": Fragments.fr_urea,
                            "Number of peroxide groups": peroxide_count,
                            "Number of ortho OH (wrt OH)": ortho_ring_count,
                            "Number of meta OH (wrt OH)": meta_ring_count,
                            "Number of para OH (wrt OH)": para_ring_count
                            }
    functional_group_calc_dataframe=pd.DataFrame()
    for key, value in functional_group_dict.items():
            functional_group_calc_dataframe["mol"] = smiles.map(MolFromSmiles)
            functional_group_calc_dataframe[key] = functional_group_calc_dataframe["mol"].map(value)
            functional_group_calc_dataframe.drop(["mol"], inplace=True, axis=1)
    return functional_group_calc_dataframe
