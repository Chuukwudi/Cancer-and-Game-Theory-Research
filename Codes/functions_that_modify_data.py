import torch
import numpy as np
import pandas as pd
import pybiomart as bm
import copy
import re
from scipy.stats import zscore

pd.options.mode.chained_assignment = None

dtype = torch.FloatTensor

genes = pd.read_csv('../more_data/genes.csv')
grRules = pd.read_excel('../more_data/grRules2.xlsx')


def get_gene(line):
    """This function takes a string (a row in this case) and searches for
    gene names and returns a list containing the found genes"""

    w = []
    # just an empty list.

    pattern = re.compile(r'ENSG\d\d\d\d\d\d\d\d\d\d\d')
    # created the pattern to be searched in the form of 'ENSGXXXXXXXXXXXXXXX'

    # find all the patterns.
    # in case the input to this function is not a string, return an empty list.
    if type(line) == str:
        found = re.findall(pattern=pattern, string=line)
        return found
    else:
        return w


# This is where the encoding takes place.
def generate_matrix(col_name):
    """This function takes the column name of the grRules encodes it into a
    dataframe and returns the encoded dataframe"""

    genes_list = genes['genes'].tolist()
    data = []
    for line in col_name.grRules:
        roo = []
        foun = get_gene(line)
        for gen in genes_list:
            if len(foun) == 0:
                roo.append(0)
            elif gen in foun:
                roo.append(1)
            else:
                roo.append(0)
        data.append(roo)
    df = pd.DataFrame(data, columns=genes_list)
    return df

# Get biomart data
dataset = bm.Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
full_genes_and_names = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
full_genes_and_names


def edit(data_path):
    '''Function converts data into ensemble header'''
    df = pd.read_csv(data_path)

    # Select relevant columns
    new_name = {}
    for column in df:
        if column in full_genes_and_names['Gene name'].values:
            replacement = full_genes_and_names.loc[full_genes_and_names['Gene name'] == str(column), 'Gene stable ID'].values[0]
            new_name[str(column)] = str(replacement)

            try:
                duplicate = full_genes_and_names.loc[full_genes_and_names['Gene name'] == str(column), 'Gene stable ID'].values[1]
                df[str(duplicate)] = df[str(column)]
            except:
                pass

    df.rename(new_name, axis=1, inplace=True)

    extra = df[['SAMPLE_ID', 'OS_MONTHS', 'OS_EVENT', 'AGE']]

    main = df.loc[:, df.columns.str.startswith('ENSG')]

    main['SAMPLE_ID'] = extra['SAMPLE_ID']
    main['OS_MONTHS'] = extra['OS_MONTHS']
    main['OS_EVENT'] = extra['OS_EVENT']
    main['AGE'] = extra['AGE']

    return main


def zscore_normalisation(data_path):
    ensemble = edit(data_path)

    extra = ensemble[['SAMPLE_ID', 'OS_MONTHS', 'OS_EVENT', 'AGE']]

    main = ensemble.drop(['SAMPLE_ID', 'OS_MONTHS', 'OS_EVENT', 'AGE'], axis=1)
    main = main.apply(zscore)
    main['SAMPLE_ID'] = extra['SAMPLE_ID']
    main['OS_MONTHS'] = extra['OS_MONTHS']
    main['OS_EVENT'] = extra['OS_EVENT']
    main['AGE'] = extra['AGE']

    return main


def discretise(data_path):
    df = zscore_normalisation(data_path)

    extra = df[['SAMPLE_ID', 'OS_MONTHS', 'OS_EVENT', 'AGE']]
    main = df.drop(['SAMPLE_ID', 'OS_MONTHS', 'OS_EVENT', 'AGE'], axis=1)

    for column in main.columns:
        main[str(column)] = np.where((main[str(column)] < 1) & (main[str(column)] > -1) , 0, 1)

    main['SAMPLE_ID'] = extra['SAMPLE_ID']
    main['OS_MONTHS'] = extra['OS_MONTHS']
    main['OS_EVENT'] = extra['OS_EVENT']
    main['AGE'] = extra['AGE']

    return main


def get_pathway(data_path):
    Human_GEM = pd.read_excel("../more_data/Human-GEM.xlsx")
    rules_and_pathway = Human_GEM[['GENE ASSOCIATION', 'SUBSYSTEM']]
    rules_and_pathway.fillna(0, inplace=True)

    variables = {}
    data = discretise(data_path)

    extra = data[['SAMPLE_ID', 'OS_MONTHS', 'OS_EVENT', 'AGE']]

    data.drop(['SAMPLE_ID', 'OS_MONTHS', 'OS_EVENT', 'AGE'], axis=1, inplace=True)

    # variables present in the rules but not in the data will be set to zero 0
    all_ensemble_genes = full_genes_and_names['Gene stable ID'].tolist()
    available_ensemble_genes = data.columns.tolist()
    for item in all_ensemble_genes:
        if item not in available_ensemble_genes:  # You might change this to a 1
            variables[item] = 0

    for items in genes['genes'].tolist():
        if items not in available_ensemble_genes:
            variables[items] = 0

    # for item in rxn_genes, if item not in available_ensemble_genes, make it zero.

    pathways = []
    for row_index in range(len(data)):
        # fetch the values
        for column in data.columns:
            value = data.loc[row_index, column]
            variables[column] = value

        # assign these values to the keys
        globals().update(variables)

        # evaluate each gene rule
        dfc = []
        active_pathways = []
        index = -1
        for line in rules_and_pathway['GENE ASSOCIATION']:
            index += 1
            outcome = eval(str(line))
            if outcome == 1:
                active_pathways.append(rules_and_pathway.loc[index, 'SUBSYSTEM'])
        a_set = set(active_pathways)
        dictt = {element: '1' for element in a_set}
        pathways.append(dictt)

    df = pd.DataFrame(pathways)
    df.fillna(0, inplace=True)

    df['SAMPLE_ID'] = extra['SAMPLE_ID']
    df['OS_MONTHS'] = extra['OS_MONTHS']
    df['OS_EVENT'] = extra['OS_EVENT']
    df['AGE'] = extra['AGE']

    return df

