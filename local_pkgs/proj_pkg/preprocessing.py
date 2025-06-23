import pandas as pd
from pymatgen.core import Composition

# global variables
REDUCED_COMP_COL = "reduced_compositions"

def reduce_comps_ls(compositions : list) -> list:
    """
    Normalize the list or array of compositions by converting each entry to its reduced formula.
    This ensures that compositions are order-independent and standardized.
    
    Parameters
    ----------
    compositions : list or array-like
        A list or array-like structure containing `pymatgen.Composition` objects or valid strings.
    
    Returns
    -------
    normalized_compositions : list of str or pymatgen.Composition
        A list of string or pymatgen.Composition representations of the reduced formulas of the compositions.

        Examples
    --------
    >>> compositions = [Composition("NaCl"), Composition("ClNa"), Composition("Na2Cl2")]
    >>> normalized = reduce_comps_ls(compositions)
    >>> print([str(comp) for comp in normalized])
    ['NaCl', 'NaCl', 'NaCl']
    """
    try:
        normalized_compositions = [
            comp.reduced_formula if isinstance(comp, Composition) else Composition(comp).reduced_formula
            for comp in compositions
        ]
        return normalized_compositions
    except Exception as e:
        raise ValueError("Invalid Composition data found. Ensure all entires are valid chemical formula.")

def reduce_comps_in_df(df, composition_column, inplace=False):
    """
    Normalize the compositions in the DataFrame by converting each entry to its reduced formula.
    This ensures that the compositions are order-independent and standardized for further grouping.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing composition data.
    
    composition_column : str
        The name of the column containing `pymatgen.Composition` or str objects.
    
    inplace : bool, optional
        If True, modifies the DataFrame directly and adds the 'normalized_composition' column in place.
        If False, returns a new DataFrame with the normalized compositions. Default is False.
    
    Returns
    -------
    pd.DataFrame
        If `inplace` is False, returns a DataFrame with an additional 'normalized_composition' column,
        containing the reduced formula of the compositions.
    
    Notes
    -----
    - The function uses `normalize_compositions()` to normalize the compositions.
    
    Examples
    --------
    >>> from pymatgen.core import Composition
    >>> import pandas as pd
    >>> data = {'composition': [Composition("NaCl"), Composition("ClNa"), Composition("Na2Cl2")]}
    >>> df = pd.DataFrame(data)
    >>> reduce_comps_in_df(df, 'composition', inplace=True)
    >>> print(df)
    """
    if not inplace:
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()

    # Apply normalize_compositions to the specified column
    df[REDUCED_COMP_COL] = reduce_comps_ls(df[composition_column])
    composition_idx = df.columns.get_loc(composition_column)
    df.insert(composition_idx + 1, REDUCED_COMP_COL, df.pop(REDUCED_COMP_COL))  # insert next to the composition column

    # Return the modified DataFrame if inplace is False
    if not inplace:
        return df

def scientific_to_numeric_compositions(composition, decimal_places=6):
    """
    Convert subscripts in a Composition from scientific (e.g. 1e-3) to numeric (e.g. 0.001) notation. This is to ensure the compositions are parsed property by some models.

    Parameters
    ----------
    composition : Composition
        The Composition object to convert.

    decimal_places : int, optional
        The maximum number of decimal places to use for the numeric notation. Default is 6.

    Returns
    -------
    str
        The Composition as a string with numeric notation for the subscripts.

    """
    if not isinstance(composition, Composition):
        try:
            composition = Composition(composition)
        except Exception as e:
            raise ValueError("Invalid Composition data found. Ensure all entires are valid chemical formula.")

    # Create a dictionary with elements and their amounts
    element_amounts = composition.get_el_amt_dict()
    # Convert each amount to a string with regular notation
    format_string = f"{{:.{decimal_places}f}}"
    converted = {
        el: format_string.format(amt).rstrip('0').rstrip('.') if amt < 1 or amt != int(amt) else str(int(amt))
        for el, amt in element_amounts.items()
    }
    # Join the elements with their amounts as a string
    return ''.join([f"{el}{amt}" for el, amt in converted.items()])

def normalize_comp(comp, return_type='composition'):
    """
    Normalize a pymatgen Composition object so the sum of atomic fractions equals 1.

    Args:
        comp (Composition or str): The pymatgen Composition object or string to normalize.
        return_type (str): The return type - 'composition' for a pymatgen Composition 
                           or 'dict' for a dictionary of normalized atomic fractions.

    Returns:
        Composition or dict: Normalized composition as specified by `return_type`.
    """
    # Convert string input to Composition, ignore case
    if isinstance(comp, str):
        comp = Composition(comp)

    if not isinstance(comp, Composition):
        raise TypeError("Input must be a pymatgen Composition object or a valid chemical formula string.")
    
    # Normalize the composition
    normalized_comp = {el: amt / comp.num_atoms for el, amt in comp.items()}
    
    if return_type.lower() == 'composition':
        return Composition(normalized_comp)
    elif return_type.lower() == 'dict':
        return normalized_comp
    else:
        raise ValueError("Invalid return_type. Use 'composition' or 'dict'.")

def normalize_comp_array(comp_array, return_type='composition'):
    """
    Normalize an array of pymatgen Composition objects or string representations.

    Args:
        comp_array (list, pd.Series, or similar): Array-like structure containing compositions.
        return_type (str): The return type - 'composition' for pymatgen Composition objects
                           or 'dict' for dictionaries of normalized atomic fractions.

    Returns:
        pd.Series: Normalized compositions as specified by `return_type`.
    """
    try:
        normalized_array = [
            normalize_comp(comp, return_type=return_type) for comp in comp_array
        ]
        return pd.Series(normalized_array, name="Normalized Composition")
    except Exception as e:
        raise ValueError(f"Error processing compositions: {e}")
