import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor


def corr_heatmap(corr):
    """
    Returns a heatmap display of the correlation between variables
    """
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    fig1, ax1 = plt.subplots(figsize=(20,15))
    sns.heatmap(corr, mask=mask, ax=ax1, cmap='viridis');
    
    
def get_predictors(df, outcome):
    """
    Return a list of predictors based on the specified outcome column of a dataset
    """
    return [c for c in df.columns if c != outcome]


def create_formula(outcome, predictors):
    """
    Return a string formula for statsmodels OLS
    """
    f = outcome + ' ~ ' + ' + '.join(predictors)
    return f


def get_vif(df, predictors):
    """
    Return a variance inflation factor dataframe for all the 
    predictors used in a model
    """
    rows = df[predictors].values
    vif_df = pd.DataFrame()
    vif_df["feature"] = predictors
    vif_df["VIF"] = [variance_inflation_factor(rows, i) for i in range(len(predictors))]
    return vif_df


def check_lr_assumptions(model, df, outcome, predictors):  
    """
    Combines 3 tests Rainbow for Linearity, Jarque-Bera and QQ-plot for Normality, 
    Breusch-Pagan test and as Residuals Plot for Homoskedasticity,
    as well as variance inflation factor for Multicollinearity
    """
    print('\t\t\t\t\tStatistic: \t\t\tP-value:')
    # Linearity (using Rainbow Test)
    rainbow_statistic, rainbow_p_value = linear_rainbow(model)
    print(f'(Linearity) Rainbow Test: \t\t{rainbow_statistic} \t\t{rainbow_p_value}')    
    # Normality (using Jarque-Bera)
    residuals = model.resid
    jb = jarque_bera(residuals, axis=0)
    jb_statistic, jb_p_value = jb[:2]
    print(f'(Normality) Jarque-Bera Test: \t\t{jb_statistic} \t\t{jb_p_value}')
    print()
    # Homoskedasticity (using Breusch-Pagan test)
    y = df[outcome]
    y_hat = model.predict()
    print('\t\t\t\t\tLagrange Multiplier p-value \tF-statistic p-value:')    
    lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-y_hat, df[predictors])
    print(f'(Homoskedasticity) Breusch-Pagan test: \t{lm_p_value} \t\t\t\t{f_p_value}')
    
    # print out vif dataframe if there are more than 1 predictors being used
    if len(predictors) >= 2:
        vif_df = get_vif(df, predictors)
        print()
        print('Independence:')
        print(vif_df)
    
    # Plot residuals to check for homoskedasticity
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    ax1.set(xlabel='Predicted Sale Price',
            ylabel='Residuals (Predicted - Actual Sale Price)')
    ax1.hlines(y=0, xmin=y_hat.min(), xmax=y_hat.max(), color='red', lw=0.5)
    ax1.scatter(x=y_hat, y=y_hat-y, color='blue', alpha=0.2);
    
    # Plot residuals using qq to check for normality
    fig = sm.qqplot(residuals, line='45', fit=True, ax=ax2)
    fig.show();
    
    
def append_results(model_df, model_name, formula, model, df, outcome, predictors, alpha=0.05, note=None):
    """
    Append model fitting results to a summary dataframe 'model_df'
    """
    r2 = model.rsquared
    
    # rainbow (for linearity)
    rainbow_statistic, rainbow_p_value = linear_rainbow(model)
    linear_eval = 'Violated' if rainbow_p_value < alpha else 'Satisfied'
    
    # jarque-bera (for normality)
    residuals = model.resid
    jb = jarque_bera(residuals, axis=0)
    jb_statistic, jb_p_value = jb[:2]
    normal_eval = 'Violated' if jb_p_value < alpha else 'Satisfied'
    
    # breusch-pagan (for hetero)
    y = df[outcome]
    yhat = model.predict()
    lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-yhat, df[predictors])
    homoskedasticity_eval = 'Violated' if f_p_value < alpha else 'Satisfied'
    
    # check for multicollinearity
    # if there's only 1 predictor, Independence Assumption is not applicable
    if len(predictors) == 1:
        multicollinearity = 'N/A'
        independence_eval = 'N/A'
    # if there're more than 1 predictors
    else:
        vif_df = get_vif(df, predictors)
        # select all the predictors with a VIF values greater or equal to 5
        # ("rule of thumb" for VIF is 5 is too high)
        multicollinearity = ', '.join(vif_df[vif_df.VIF >= 5].feature.values)
        independence_eval = 'Violated' if multicollinearity != '' else 'Satisfied'
    
    # append results to dataframe
    model_df.loc[len(model_df)] = [model_name, formula, r2, 
                                   rainbow_statistic, rainbow_p_value, linear_eval,
                                   jb_statistic, jb_p_value, normal_eval, 
                                   lm_p_value, f_p_value, homoskedasticity_eval,
                                   multicollinearity, independence_eval,
                                   note]
    return model_df


def remove_outliers_z(dt, col, threshold=3):
    """
    Remove outliers of a dataframe 'dt' using z-score for specified column 'col'
    By default, z-score thresholds are +/- 3 for a normal distribution 
    """
    z_scores = stats.zscore(dt[col])
    abs_z_scores = np.abs(z_scores)
    output = dt[abs_z_scores < threshold]
    return output


def remove_outliers_iqr(dt, col):
    """
    Remove outliers of a dataframe 'dt' based on IQR specified column 'col'    
    """
    q1 = dt[col].quantile(0.25)
    q3 = dt[col].quantile(0.75)
    iqr = q3 - q1
    output = dt[(dt[col] > q1 - 1.5 * iqr) & (dt[col] < q3 + 1.5 * iqr)]
    return output


def log_transform(cols, dt):
    '''
    Applies a log transformation that also prefixes 'log_' to column names
    '''
    output = dt.copy()
    for c in dt.columns:
        if c in cols:
            output[f'log_{c}'] = np.log(dt[c])
            output.drop(c, axis=1, inplace=True)
    return output


def sq_rt(cols, dt):
    '''
    Applies a square root transformation that also prefixes 'sqrt_' to column names
    '''
    output = dt.copy()
    for c in dt.columns:  
        if c in cols:
            output.drop(c, axis=1, inplace=True)
            output[f'sqrt_{c}'] = (dt[c])**.5
    return output


def get_numeric_features(df):
    '''
    Return all columns of type 'int64' or 'float64'
    '''
    numeric_features = []
    for c in df.columns:
        if (df[c].dtype == 'int64' or df[c].dtype == 'float64') and c != 'SalePrice':
            numeric_features.append(c)
    return numeric_features


def scatterplot_feats_outcome(df, features, outcome):
    """
    Visualizing the relationship between all numeric_features with target variable 'outcome' 
    using scatter plots in the dataframe 'df'
    """
    nrows = len(features) // 3
    nrows += 1 if (len(features) % 3 != 0) else 0
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(21,nrows*5))
    for idx, f in enumerate(features):
        ax = axes[idx // 3][idx % 3]
        sns.scatterplot(data=df, x=outcome, y=f, ax=ax, color='blue', alpha=0.4)
    fig.tight_layout()
    fig.show();
    
    
def convert_to_binary(x):
    '''
    Converts x to binary 0/1
    Output would be similar to LabelEncoder
    '''
    return 1 if x > 0 else 0


def reformat_col_names(df):
    '''Removes all white spaces, and '.' characters, and replaces all '/' and '-' with '_'
    in column names to avoid problems with creating R formula later on
    '''
    df.columns = [x.replace(" ", '')
                   .replace('.', '')
                   .replace("-", "_")
                   .replace("/", "_") for x in df.columns]
    
    
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ 
    Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()    # should be .idxmax(), not .argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def forward_selection(X, y, 
                      initial_list=[],  
                      verbose=True):
    """ 
    Perform a forward feature selection 
    based on Adjusted R-Squared from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    
    """
    included = list(initial_list)
    excluded = list(set(X.columns) - set(included))
    best_adj_r2 = 0.0
    current_adj_r2 = 0.0
    while (len(excluded) > 0):
        increased = False
        col_adj_r2 = []
        for column in X.columns:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[column]]))).fit()
            col_adj_r2.append((column, model.rsquared_adj))
            
        # sort temp_dct by values 
        col_adj_r2.sort(key=lambda x: x[1], reverse=True)
        best_col, best_adj_r2 = col_adj_r2[0]

        if current_adj_r2 < best_adj_r2:
            increased = True
            current_adj_r2 = best_adj_r2
            included.append(best_col)
            excluded = list(set(X.columns) - set(included))
            if verbose:
                print(f'Add: {best_col:30} Current Best Adjusted R2: {current_adj_r2:.6}')
            print(len(included), len(excluded))
        if not increased:
            break

    return included


def find_features(keyword, params):
    '''
    Retrieves features which contain certain keyword in their names
    and their coefficients from a pd Series of params 
    '''
    return params[params.index.str.contains(keyword)]


def identify_multicollinearity(feature, df):
    '''
    Identifies other features that are highly correlated to specified input feature
    using a threshold of +/- 0.5
    '''
    corr = df.corr()[feature].sort_values(ascending=False)
    output = corr[(np.abs(corr) > 0.5)]
    return output

def regionalize(x):
    '''
    Categorizes input (district names in King County)
    into regions (North, East, South, Seattle)
    Reference: https://www.communitiescount.org/king-county-geographies
    '''
    north = [x.upper() for x in ['Bothell', 'Cottage Lake', 'Kenmore', 'Lake Forest Park', 
                                 'Shoreline', 'Woodinville']]
    east = [x.upper() for x in ['Bellevue', 'Carnation', 'Duvall', 'Issaquah', 'Kirkland', 
                                'Medina', 'Mercer Island', 'Newcastle', 'North Bend', 
                                'Redmond', 'Sammamish', 'Skykomish', 'King County', 
                                'SNOQUALMIE', 'BEAUX ARTS', 'CLYDE HILL', 'YARROW POINT',
                                'HUNTS POINT']]

    south = [x.upper() for x in ['Auburn', 'Burien', 'Covington', 'Des Moines', 'Enumclaw', 
                                 'Federal Way', 'Kent', 'Maple Valley', 'Normandy Park', 
                                 'Renton', 'Tukwila', 'White Center', 'Boulevard Park', 
                                 'Vashon Island', 'Algona', 'Pacific', 'BLACK DIAMOND',
                                 'MILTON', '']] + ['SeaTac']
    seattle = ['SEATTLE']

    regions = [north, south, east, seattle]
    for region in regions:
        if x in north:
            output = 'North'
        elif x in east:
            output = 'East'
        elif x in south:
            output = 'South'
        elif x in seattle:
            output = 'Seattle'
    return output

def clean_col_names(df):
    """
    Remove empty white spaces " ", and "." character,
    replace "-" and "/" with "_" in column names of a dataframe "df"
    """
    df.columns = [x.replace(" ", '')
                   .replace('.', '')
                   .replace("-", "_")
                   .replace("/", "_") for x in df.columns]

    
def separate(dt, col, target_val, threshold=0):
    """
    Separate dataframe 'dt' based threshold value 'threshold' for column 'col'
    Collect 'df[target_val]' into 2 different arrays
    """
    gt = dt[dt[col] > threshold][target_val].values
    le = dt[dt[col] <= threshold][target_val].values
    
    return gt, le