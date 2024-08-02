import pandas as pd


def extraction_donnees(path, sheet, usecols, first_row):
    noms = pd.read_excel(path, sheet_name=sheet, usecols=usecols, skiprows=first_row - 3, nrows=1)
    numpy_noms = noms.to_numpy()

    donnes = pd.read_excel(path, sheet_name=sheet, usecols=usecols, skiprows=first_row - 1)
    donnes = donnes.fillna(0)
    numpy_donnees = donnes.to_numpy()

    return numpy_donnees, numpy_noms


path = "C:/Users/garan/Documents/Ecole/L3/Stage L3/Code/bus/donnees/LA_JOB.xlsx"
sheet = "LAS2_trhor15=t_0845-0859"
usecols = 'C:Q'

if __name__ == "__main__":
    donnes, noms = extraction_donnees(path, sheet, usecols, 7)
    print(donnes)
    print(noms)
