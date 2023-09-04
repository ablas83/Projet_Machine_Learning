from models.model import dt_param_selector
import numpy as np
def getAlgorims(df):
    classification = {'DecisionTreeClassifier': dt_param_selector}
    regression = {'Rasso': dt_param_selector}
    column_type = df['target'].dtype
    if column_type == 'object':
        return classification
    else:
        return regression
