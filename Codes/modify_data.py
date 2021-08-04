import os.path
from functions_that_modify_data import *

# Get matrix from rules
# Save matrix
if os.path.exists('../data/new_entire_data.csv'):
    pass
else:
    validation_pathway = get_pathway('../data/validation.csv')
    validation_pathway.to_csv('../data/new_validation.csv', index=False)

    train_pathway = get_pathway('../data/train.csv')
    train_pathway.to_csv('../data/new_train.csv', index=False)

    test_pathway = get_pathway('../data/test.csv')
    test_pathway.to_csv('../data/new_test.csv', index=False)

    entire_data_pathway = get_pathway('../data/entire_data.csv')
    entire_data_pathway.to_csv('../data/new_entire_data.csv', index=False)
