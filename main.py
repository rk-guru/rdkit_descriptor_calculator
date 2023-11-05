import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Descriptors3D, Lipinski
from rdkit.Chem import rdMolDescriptors, GraphDescriptors, Fragments
from descriptor_list import *
# from self_functions import *
import warnings
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

def Descriptor(smile_list):
    smiles_list = smile_list
    dataframes = [
        rd_MolDescriptors(smiles_list),
        descriptors(smiles_list),
        mole_property_calc(smiles_list),
        atomic_property_calc(smiles_list),
        functional_group_calc(smiles_list),
        lipinski(smiles_list),
        graph_descp(smiles_list),
    ]

    pd.options.display.width = None
    Smile_descriptors = pd.concat(
        dataframes,
        keys=["rd_MolDescriptors","Descriptors",'mole_property_df',' bond_property_df', 'atomic_property_df',' functional_group_df',
            "Lipinski",
            "Graph Descriptors",
        ],
        axis=1,
    )
    print(Smile_descriptors)
    return Smile_descriptors