from models.model import dt_param_selector

def getAlgorims(df):
    classification = [('Decision Tree', dt_param_selector())]
    regression = [('rasso', dt_param_selector())]
    if isinstance(df['target'], str):
        return classification
    else:
        return regression
