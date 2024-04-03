# POFP
The POFP.py contains python functions to generate photo-oriented fingerprints.
## Dependencies
- Numpy
- RDkit

## `GenSpecFP` Function

#### Description
The `GenSpecFP` function generates a specific fingerprint for a given molecule. The fingerprint is a bit vector based on substructure matching within the molecule and can be used for molecular similarity assessment in cheminformatics and drug design.

#### Parameters
- `mol`: A string representing the molecule's SMILES (Simplified Molecular Input Line Entry System) notation. SMILES is a linear notation for describing the structure of chemical substances.
- `mode`: An integer that specifies the mode of fingerprint generation.
  - `mode=1`: The corresponding bit in the fingerprint is set to 1 if the molecule contains a specific substructure; otherwise, it is set to 0.
  - `mode=2`: The corresponding bit in the fingerprint is set to the number of occurrences of a specific substructure in the molecule.
- `with_aromaticity` (optional): A boolean indicating whether to include aromaticity information in the fingerprint. Defaults to `True`.

#### Return Value
- Returns `np.nan` if the input SMILES string cannot be parsed into a valid molecule.
- Otherwise, returns a NumPy array representing the molecule's specific fingerprint. Each element of the array corresponds to the presence (`mode=1`) or the number of occurrences (`mode=2`) of specific substructures. If `with_aromaticity` is `True`, the last element of the array represents the number of aromatic rings in the molecule.

#### Exceptions
- The function may raise an exception if the `mol` parameter is not a valid SMILES string.
