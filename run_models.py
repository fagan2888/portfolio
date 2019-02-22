"""
Generate data for paper

"""

import os, sys
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt
import copy

# Add the module to source path
dirname = os.path.dirname(os.path.realpath(__file__))
source_path = dirname + "/portfolio"
sys.path.append(source_path)

from portfolio.model import SingleMethod, LinearModel#, Markowitz, NN

#def calc_mean_and_var(x):
#    vars_ = np.var(x, ddof = 1, axis = 1)
#    means = np.mean(x, axis = 1)
#    alphas = vars_ / sum(vars_)
#    mean = sum(means * alphas)

#def run_SingleMethod(x,y, names = None, classes = None):
#    if os.path.isfile("pickles/single_method_results.pkl"):
#        print("single method results already generated")
#        return
#
#
#    m = SingleMethod()
#    cv_params = {'loss': ('mae', 'rmsd', 'max')}
#    # Separate different classes and methods
#    unique_classes = np.unique(classes)
#    unique_costs = np.unique(cost)
#
#    d = {}
#
#    for cl in unique_classes:
#        params = []
#        estimators = []
#        errors = []
#        cv_portfolios = []
#        portfolios = []
#        portfolio_weights = []
#        portfolio_names = []
#        maes = []
#        rmsds = []
#        maxerrs = []
#        sample_maes = []
#        sample_rmsds = []
#        laplace_mnll = []
#        gaussian_mnll = []
#        for co in unique_costs:
#            class_idx = np.where(classes == cl)[0]
#            cost_idx = np.where(cost <= co)[0]
#            # Get best hyperparams from cv
#            error, portfolio, best_params, cv_portfolio = \
#                    outer_cv(x[np.ix_(class_idx, cost_idx)], y[class_idx], m, cv_params)
#            # Train model with best hyperparams on full data
#            m.set_params(**best_params)
#            m.fit(x[np.ix_(class_idx, cost_idx)], y[class_idx])
#            # Get human readable details on the selected portfolio
#            portfolio_weight, portfolio_name = get_portfolio_details(portfolio, names)
#            # Get statistics
#            mae, rmsd, maxerr, sample_mae, sample_rmsd, nll_laplace, nll_gaussian = get_error_statistics(error)
#
#            # Store various attributes for the given cost
#            params.append(best_params)
#            estimators.append(copy.copy(m))
#            errors.append(error)
#            cv_portfolios.append(cv_portfolio)
#            portfolios.append(portfolio)
#            portfolio_weights.append(portfolio_weight)
#            portfolio_names.append(portfolio_name)
#            maes.append(mae)
#            rmsds.append(rmsd)
#            maxerrs.append(maxerr)
#            sample_maes.append(sample_mae)
#            sample_rmsds.append(sample_rmsd)
#            laplace_mnll.append(nll_laplace)
#            gaussian_mnll.append(nll_gaussian)
#
#        # Store various attributes for the given class
#        d[cl] = {'errors': errors,
#                 'cv_portfolios': cv_portfolios,
#                 'portfolios': portfolios,
#                 'params': params,
#                 'estimators': estimators,
#                 'cost': unique_costs,
#                 'portfolio_weights': portfolio_weights,
#                 'portfolio_names': portfolio_names,
#                 'maes': maes,
#                 'rmsds': rmsds,
#                 'maxerrs': maxerrs,
#                 'sample_maes': sample_maes,
#                 'sample_rmsds': sample_rmsds,
#                 'laplace_mnll': laplace_mnll,
#                 'gaussian_mnll': gaussian_mnll
#                 }
#
#    # Dump the results in a pickle
#    with open("pickles/single_method_results.pkl", 'wb') as f:
#        pickle.dump(d, f, -1)

#def get_error_statistics(errors):
#    mae = np.mean(abs(errors))
#    rmsd = np.sqrt(np.mean(errors**2))
#    maxerr = np.max(abs(errors))
#
#    sample_mae = np.zeros(errors.size)
#    sample_rmsd = np.zeros(errors.size)
#    for idx, j in sklearn.model_selection.LeaveOneOut().split(errors):
#        sample_mae[j] = np.mean(abs(errors[idx]))
#        sample_rmsd[j] = np.sqrt(np.mean(errors[idx]**2))
#
#    nll_laplace = np.mean(np.log(2*sample_mae) + abs(errors) / sample_mae)
#    nll_gaussian = 0.5 * np.log(2 * np.pi) + np.mean(np.log(sample_rmsd) + errors**2 / sample_rmsd**2)
#
#    return  mae, rmsd, maxerr, sample_mae, sample_rmsd, nll_laplace, nll_gaussian


#def run_LinearModel(x,y, cost = None, names = None, classes = None):
#    if os.path.isfile("pickles/linear_method_results.pkl"):
#        print("Linear method results already generated")
#        return
#
#
#    m = NN(tensorboard_dir = '', learning_rate = 1e-1, iterations = 20000, 
#            l2_reg = 1e-6, cost_reg = 0, cost = cost) # cost_reg ~ (1e-6, 1e5) good start
#
#    cost_reg = [0] + [10**i for i in range(-6,5)]
#    m.fit(x,y)
#    idx = np.argsort(m.portfolio)[-5:]
#    print(m.portfolio[idx], sum(m.portfolio > 1e-6))
#    quit()
#    cv_params = {'loss': ('mae', 'rmsd', 'max')}
#    # Separate different classes and methods
#    unique_classes = np.unique(classes)
#    unique_costs = np.unique(cost)
#
#    d = {}
#
#    for cl in unique_classes:
#        params = []
#        estimators = []
#        errors = []
#        cv_portfolios = []
#        portfolios = []
#        portfolio_weights = []
#        portfolio_names = []
#        maes = []
#        rmsds = []
#        maxerrs = []
#        sample_maes = []
#        sample_rmsds = []
#        laplace_mnll = []
#        gaussian_mnll = []
#        for co in unique_costs:
#            class_idx = np.where(classes == cl)[0]
#            cost_idx = np.where(cost <= co)[0]
#            # Get best hyperparams from cv
#            error, portfolio, best_params, cv_portfolio = \
#                    outer_cv(x[np.ix_(class_idx, cost_idx)], y[class_idx], m, cv_params)
#            # Train model with best hyperparams on full data
#            m.set_params(**best_params)
#            m.fit(x[np.ix_(class_idx, cost_idx)], y[class_idx])
#            # Get human readable details on the selected portfolio
#            portfolio_weight, portfolio_name = get_portfolio_details(portfolio, names)
#            # Get statistics
#            mae, rmsd, maxerr, sample_mae, sample_rmsd, nll_laplace, nll_gaussian = get_error_statistics(error)
#
#            # Store various attributes for the given cost
#            params.append(best_params)
#            estimators.append(copy.copy(m))
#            errors.append(error)
#            cv_portfolios.append(cv_portfolio)
#            portfolios.append(portfolio)
#            portfolio_weights.append(portfolio_weight)
#            portfolio_names.append(portfolio_name)
#            maes.append(mae)
#            rmsds.append(rmsd)
#            maxerrs.append(maxerr)
#            sample_maes.append(sample_mae)
#            sample_rmsds.append(sample_rmsd)
#            laplace_mnll.append(nll_laplace)
#            gaussian_mnll.append(nll_gaussian)
#
#        # Store various attributes for the given class
#        d[cl] = {'errors': errors,
#                 'cv_portfolios': cv_portfolios,
#                 'portfolios': portfolios,
#                 'params': params,
#                 'estimators': estimators,
#                 'cost': unique_costs,
#                 'portfolio_weights': portfolio_weights,
#                 'portfolio_names': portfolio_names,
#                 'maes': maes,
#                 'rmsds': rmsds,
#                 'maxerrs': maxerrs,
#                 'sample_maes': sample_maes,
#                 'sample_rmsds': sample_rmsds,
#                 'laplace_mnll': laplace_mnll,
#                 'gaussian_mnll': gaussian_mnll
#                 }
#
#    # Dump the results in a pickle
#    with open("pickles/single_method_results.pkl", 'wb') as f:
#        pickle.dump(d, f, -1)

def parse_reaction_dataframe(df):
    # just to make sure that stuff is sorted
    # supress warning as this works like intended
    pd.options.mode.chained_assignment = None
    # Sort to make it simpler to get the data in the right format
    df.sort_values(['reaction','name'], inplace=True)
    pd.options.mode.chained_assignment = "warn"

    # Create dataframe with subset not including the reference
    # as well as one that only includes the reference
    df_ref = df.loc[df.functional == 'uCCSD']
    df_noref = df.loc[df.functional != 'uCCSD']
    unique_reactions = df_noref.reaction.unique()
    n = unique_reactions.size
    unique_names = df_noref.name.unique()
    m = unique_names.size
    # Make dataframe containing a single reaction for getting
    # properties that are constant across reactions
    df_reaction = df_noref.loc[df_noref.reaction == unique_reactions[0]]
    # Make dataframe containing a single method for getting
    # properties that are constant across methods
    df_name = df_noref.loc[df_noref.name == unique_names[0]]

    # Get all the data from the dataframe and reshape them.
    energy = df_noref.energy.values.reshape(n, -1).astype(float)
    reference_energy = df_ref.energy.values.astype(float)
    basis = df_reaction.basis.values
    charge = df_name.charge.values.astype(int)
    correlation_energy = df_noref.correlation_energy.values.reshape(n, -1).astype(float)
    reference_correlation_energy = df_ref.correlation_energy.values.astype(float)
    functional = df_reaction.functional.values
    method_name = df_reaction.name.values
    one_electron_energy = df_noref.one_electron_energy.values.reshape(n, -1).astype(float)
    reaction_name = df_name.reaction.values
    reaction_class = df_name.reaction_class.values.astype(int)
    spin = df_name.spin.values.astype(int)
    two_electron_energy = df_noref.two_electron_energy.values.reshape(n, -1).astype(float)
    unrestricted = df_reaction.unrestricted.values.astype(bool)

    # One-hot encode categorical data
    encoder = OneHotEncoder(sparse=False)
    basis_one_hot = encoder.fit_transform(basis.reshape(-1, 1))
    functional_one_hot = encoder.fit_transform(functional.reshape(-1, 1))
    spin_one_hot = encoder.fit_transform(spin.reshape(-1, 1))

    name_lookup = {
                'energy': energy,
                'reference_energy': reference_energy,
                'basis': basis,
                'charge': charge,
                'correlation_energy': correlation_energy,
                'reference_correlation_energy': reference_correlation_energy,
                'functional': functional,
                'method_name': method_name,
                'one_electron_energy': one_electron_energy,
                'reaction_name': reaction_name,
                'reaction_class': reaction_class,
                'spin': spin,
                'two_electron_energy': two_electron_energy,
                'unrestricted': unrestricted,
                'basis_one_hot': basis_one_hot,
                'functional_one_hot': functional_one_hot,
                'spin_one_hot': spin_one_hot
                }

    return name_lookup

#def get_portfolio_details(x, names):
#
#    # Get the order by contribution of the portfolio
#    idx = np.argsort(x)
#
#    w = []
#    n = []
#
#    for i in idx:
#        weight = x[i]
#        if weight < 1e-9:
#            continue
#        w.append(weight)
#        n.append(names[i])
#
#    return w, n

#def outer_cv(x, y, m, params, grid = True, 
#        outer_cv_splits = 3, outer_cv_repeats = 1, inner_cv_splits = 3, inner_cv_repeats = 1):
#    """
#    Do outer cross validation to get the prediction errors of a method. 
#    """
#
#    if grid:
#        cv_model = sklearn.model_selection.GridSearchCV
#    else:
#        cv_model = sklearn.model_selection.RandomizedSearchCV
#
#    outer_cv_generator = sklearn.model_selection.RepeatedKFold(
#            n_splits = outer_cv_splits, n_repeats = outer_cv_repeats)
#    inner_cv_generator = sklearn.model_selection.RepeatedKFold(
#            n_splits = inner_cv_splits, n_repeats = inner_cv_repeats)
#
#    best_cv_params = []
#    cv_portfolios = []
#    errors = np.zeros((x.shape[0], outer_cv_repeats))
#
#    for i, (train_idx, test_idx) in enumerate(outer_cv_generator.split(y)):
#        train_x, train_y = x[train_idx], y[train_idx]
#        test_x, test_y = x[test_idx], y[test_idx]
#        if len(params) > 0:
#            cvmod = cv_model(m, param_grid = params, scoring = 'neg_mean_absolute_error',
#                    return_train_score = False, cv = inner_cv_generator)
#            cvmod.fit(train_x, train_y)
#            cv_portfolios.append(cvmod.best_estimator_.portfolio)
#            best_cv_params.append(cvmod.best_params_)
#            y_pred = cvmod.predict(test_x)
#        else:
#            m.fit(train_x, train_y)
#            cv_portfolios.append(m)
#            best_cv_params.append(params)
#            y_pred = m.predict(test_x)
#
#        errors[test_idx, i // outer_cv_splits] = test_y - y_pred
#
#    # Get the best params
#    best_params = get_best_params(best_cv_params)
#
#    # retrain the model on the full data
#    m.set_params(**best_params)
#    m.fit(x, y)
#    final_portfolio = m.portfolio
#
#    # Since the repeats of the same sample is correlated, it's
#    # probably better to take the mean of them before doing
#    # summary statistics.
#    # This means that we're actually using slightly more than (m-1)/m
#    # of the data, where m is the number of CV folds.
#    # But since we're not doing learning curves, this should be fine.
#    errors = np.mean(errors, axis = 1)
#
#    cv_portfolios = np.asarray(cv_portfolios)
#
#    return errors, final_portfolio, best_params, cv_portfolios

#def get_best_params(params):
#    """
#    Attempts to get the best set of parameters
#    from a list of dictionaries containing the
#    parameters that resulted in the best cv score
#    """
#
#    # Preprocess a bit
#    d = {}
#    for param in params:
#        for key,value in param.items():
#            if key not in d: d[key] = []
#            d[key].append(value)
#
#    # Select the most likely or median value
#    for key,value in d.items():
#        # if list choose most common
#        if isinstance(value[0], str):
#            best_value = max(set(value), key=value.count)
#            d[key] = best_value
#            continue
#
#        # if numeric return median
#        d[key] = np.median(value)
#
#    return d

#TODO write predefined splits

if __name__ == "__main__":
    df = pd.read_pickle("pickles/combined_reac.pkl")
    # all but x,y is optional
    data = parse_reaction_dataframe(df)
    print(data['reaction_name'])

    #m = LinearModel(clip_value=0, l2_reg=0, positive_constraint=True, integer_constraint=False)
    #x = data['energy']
    #y = data['reference_energy']
    #m.fit(x,y)
    quit()
    #z, w = outer_cv(x,y,m,{}, True, 5, 5, 5, 1)[:2]
    #print(w[(w>1e-3) | (w<-1e-3)])
    #print(names[w.argmax()])
    #print(np.mean(abs(z)))
    #quit()
    #m = NN(tensorboard_dir = '', learning_rate = 0.1, iterations = 50000,
    #        l2_reg = 0, cost_reg = 0, cost = cost)
    #m.fit(x,y)
    #print(np.sort(m.portfolio)[-5:])
    #print(names[np.argmax(m.portfolio)])

    ##m = NN(tensorboard_dir = 'log', learning_rate = 0.001, iterations = 1000000, 
    ##        l2_reg = 1e-6, cost_reg = 1e-9, cost = cost)
    ##m.fit(x,y)
    ##quit()
    ##p = np.where(m.portfolio > 0.01)[0]
    ##out = df[df.reaction == df.iloc[df.time.idxmax()].reaction].values[p][:,[0,3,-2]]
    ##print(np.concatenate([out,m.portfolio[p,None]], axis=1))

    ##run_SingleMethod(x,y, cost, names, rclass)
    ##run_linear(x,y, cost, names, None)

    ##def __init__(self, learning_rate = 0.3, iterations = 5000, cost_reg = 0.0, l2_reg = 0.0, 
    ##        scoring_function = 'rmse', optimiser = "Adam", softmax = True, fit_bias = False,
    ##        nhl = 0, hl1 = 5, hl2 = 5, hl3 = 5, multiplication_layer = False, activation_function = "sigmoid",
    ##        bias_input = False, n_main_features = -1, single_thread = True, tensorboard_dir = '', 
    ##        tensorboard_store_frequency = 100, **kwargs):
