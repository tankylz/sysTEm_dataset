# Systematically Verified Thermoelectric (`sysTEm`) Dataset 

## Description
This repository contains the Systematically Verified Thermoelectric (`sysTEm`) Dataset, which is a compilation of experimental thermoelectric (TE) data, comprising information on the composition, temperature and transport properties. By openly sharing this dataset, we hope to advance informatics research in TE materials, in the hopes of discovering higher performance TE materials.

Most of the data were extracted from experimental works using [WebPlotDigitizer](https://automeris.io). Additionally, the `sysTEm` is formed from a merger of previous works, namely an updated version of the Materials Research Laboratory (MRL) dataset ([original work](https://doi.org/10.1021/cm400893e), [the updated dataset](https://doi.org/10.5281/zenodo.15365344)) and the Experimentally Synthesized Thermoelectric Materials (ESTM) dataset ([paper](https://doi.org/10.1038/s41524-022-00897-2
)).

Aside from manual validation methods, the `sysTEm` dataset was validated systematically, the chief of which utilizing the $zT$ formula (below), with a tolerance of 10\% between the calculated value and the extracted value.

$$
zT = \frac{\sigma S^2 T}{\kappa} = \frac{\sigma S^2 T}{\kappa_e + \kappa_l} = \frac{\text{PF}}{\kappa} T
$$

Aside from materials screening and discovery, `sysTEm` dataset can serve as a benchmarking dataset for models trained on other datasets. Those wishing to extend the TE dataset may also find the code used to validate the data, shared in this repository as well, useful. The full methodology can be found in this paper:
- [ ] TODO: add link to paper


## How to Cite
If this dataset and accompanying code has been useful for your work, please consider citing the paper and `GitHub` repo:

- [ ] TODO: add citation and bibtex for sysTEm dataset paper and dataset DOI


## `sysTEm` Dataset Columns
For the final dataset (`sysTEm_dataset.xlsx`), the following columns are presented as follows:

| Column Name                                | Data Type | Description                                                                                     |
|--------------------------------------------|-----------|-------------------------------------------------------------------------------------------------|
| `#`                                        | int       | Unique identifier assigned to each row in the dataset                                           |
| `Initial Dataset`                          | string    | Dataset of origin (e.g., This Work, extended MRL, This Work)                                                  |
| `Source Paper`                             | string    | DOI or hyperlink to the source publication                                                      |
| `Pymatgen Composition`                     | string    | Composition parsed using `pymatgen.Composition`                                                 |
| `reduced_compositions`                     | string    | Simplified formula in lowest whole-number ratio (e.g., Sb2Si2Te6 → SiSbTe3)                     |
| `Pretty Formula`                           | string    | Formula extracted from the source paper, with slight formatting to make the conversion to `pymatgen.Composition` from regex easier                                                |
| `Type of Formula`                          | string    | Indicates whether the formula is `Stoichiometric` (e.g, Ag2Se) or `Mixed Formula` (e.g, Te + 0.1 wt% InP3)                          |
| `Year`                                     | int       | Year of publication of the source paper                                                         |
| `Temperature (K)`                          | float     | Measurement temperature in Kelvin                                                               |
| `Electrical Conductivity (S/cm)`           | float     | Electrical conductivity σ, in S/cm             |
| `Seebeck Coefficient (µV/K)`               | float     | Seebeck coefficient S, in µV/K                                                                  |
| `Power Factor (µW/cmK²)`                   | float     | Power factor PF, in µW/cm·K²                                                    |
| `zT`                                       | float     | Dimensionless figure of merit                                           |
| `Total Thermal Conductivity (W/mK)`        | float     | Total thermal conductivity κ = κₑ + κₗ, in W/m·K                                                |
| `Lattice Thermal Conductivity (W/mK)`      | float     | Lattice contribution to thermal conductivity κₗ, in W/m·K                                       |
| `Electronic Thermal Conductivity (W/mK)`   | float     | Carrier contribution to thermal conductivity κₑ, in W/m·K                                   |

## How to use the dataset

You may either download the file, `sysTEm_dataset.xlsx`, which contains the final dataset, or clone the entire repository.

Formatted in `.xlsx` format, `sysTEm` is presented as a data table, making it easy to work with existing libraries. Here, we present one example loading the data for further use.

### Loading the Dataset in Python

```python
# accurate as of python 3.10.15, pandas 1.5.3, pymatgen 2024.5.1 on M3 Pro MacBook 
import pandas as pd
from pymatgen.core import Composition

df = pd.read_excel('sysTEm_dataset.xlsx') # load the dataset

# convert composition string into pymatgen.core.composition object
composition_col = 'Pymatgen Composition' # or change this to 'reduced_compositions' for reduced formula
df[composition_col] = df[composition_col].map(lambda x: Composition(x))

# continue by featurizing the composition and temperature...
```

## Using the code in this repository

You are recommended to clone the entire repository. The `local_pkgs` folder contains most of the functions used for validating the data and generating plots.

### Downloading the required dependencies
The work originally used `conda` to install the required dependencies. Here, we also share the installation method using `venv`. In the event of issues with installation, you may refer to the `full_dependencies.txt` to see what are the installed packages that we ran (on a M3 Pro MacBook).

If you are using `conda`, refer to `environment.yml` for the installation steps; if you are using `venv`, refer to the `requirements.txt` for the installation instructions.

### Raw Data and Intermediate Work
Important data files can be found in the `dataset_checkpoints` folder.

The initial extended MRL and ESTM datasets are in the folder with the prefix `'original_'`, along with intermediate forms of the merged dataset. These intermediate forms are saved at checkpoints starting from `01` to `04`.

The `manually_removed_indices.xlsx` file indicates which entries, identified by a unique integer in `#`, were removed along with the reasons for their removal.

Additionally, `walkthrough.ipynb` details the processes used to generate the final dataset, and also the figures for the paper. A copy of the figures are given in the `figures` folder.