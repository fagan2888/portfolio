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
import re

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


def run_SingleMethod(data):
    model = SingleMethod()
    cv_params = {'loss': ('mae', 'rmsd', 'max')}
    # Have to be specific with splitting, due to high correlations etc.
    outer_splits, inner_splits = leave_one_out_cv(data, include_other_reaction_types=False)
    run_method(data, model, cv_params, "pickles/single_method_same_reactions_results.pkl", outer_splits, inner_splits)
    outer_splits, inner_splits = leave_one_out_cv(data, include_other_reaction_types=True)
    run_method(data, model, cv_params, "pickles/single_method_all_reactions_results.pkl", outer_splits, inner_splits)

# TODO fix l2_reg to l1_reg
def run_LinearModel(data):
    model = LinearModel(l2_reg = 1e2, positive_constraint=False, clip_value=1e-4)
    cv_params = {}
    outer_splits, inner_splits = leave_one_out_cv(data, include_other_reaction_types=False)
    run_method(data, model, cv_params, "pickles/test.pkl", outer_splits, inner_splits)


def run_method(data, model, cv_params, path, outer_splits, inner_splits):
    """
    Run all crossvalidation etc. needed to get results for
    a method. Store everything in a pickle.
    """
    if os.path.isfile(path):
        print("%s already generated" % path)
        return

    # Create the method subsets
    subsets = [
               np.where((data['basis'] == 'sto-3g') & (data['functional'] != 'df-lmp2') & (data['functional'] != 'DCSD'))[0],
               np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P')) & (data['functional'] != 'df-lmp2') & (data['functional'] != 'DCSD'))[0],
               np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p')) & (data['functional'] != 'df-lmp2') & (data['functional'] != 'DCSD'))[0],
               np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p') | (data['basis'] == 'avdz')) & (data['functional'] != 'df-lmp2') & (data['functional'] != 'DCSD'))[0],
               np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p') | (data['basis'] == 'avdz') | (data['basis'] == 'tzvp')) & (data['functional'] != 'df-lmp2') & (data['functional'] != 'DCSD'))[0],
               np.where((data['basis'] != 'qzvp') & (data['functional'] != 'df-lmp2') & (data['functional'] != 'DCSD'))[0],
               np.where((data['functional'] != 'df-lmp2') & (data['functional'] != 'DCSD'))[0],
               np.asarray(range(data['basis'].size))
              ]

    x = data['energy']
    y = data['reference_energy']

    d = {}

    errors = []
    cv_portfolios = []
    all_cv_params = []
    for subset in subsets:
        # Get best hyperparams from cv
        error, cv_portfolio, best_cv_params = \
                outer_cv(x[:,subset], y, model, cv_params, outer_splits, inner_splits)

        # Convert the portfolios to the full set
        portfolio = np.zeros(x.shape, dtype=float)
        for i in range(x.shape[0]):
            portfolio[i,subset] = cv_portfolio[i]

        # Store various attributes for the given cost
        errors.append(error)
        cv_portfolios.append(portfolio)
        all_cv_params.append(best_cv_params)
        for i in range(len(error)):
            print(abs(error[i]), data['reaction_name'][i])

        reaction_idx = np.where(data['reaction_name'] == "CH3+H2->TS3")[0]
        dy = data['energy']-data['reference_energy'].reshape(-1,1)
        idx = np.where(abs(dy[reaction_idx, subset]) > 100)[0]
        print(data['method_name'][subset][idx])
        print(np.mean(abs(error)))
        print(get_best_params_and_model(x[:,subset], y, model, best_cv_params, data['method_name'][subset])[2:])

    d['errors'] = np.asarray(errors)
    d['cv_portfolios'] = np.asarray(cv_portfolios)
    d['cv_params'] = all_cv_params
    d['reaction_names'] = data['reaction_name']
    d['method_names'] = data['method_name']

    # Dump the results in a pickle
    with open(path, 'wb') as f:
        pickle.dump(d, f, -1)

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
#    return mae, rmsd, maxerr, sample_mae, sample_rmsd, nll_laplace, nll_gaussian


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
    # Merge heavy atom and hydrogen into same classes, meaning
    # class 5 is merged into 2 and 6 into 4
    reaction_class = df_name.reaction_class.values.astype(int)
    reaction_class[np.where(reaction_class == 5)[0]] = 2
    reaction_class[np.where(reaction_class == 6)[0]] = 4
    spin = df_name.spin.values.astype(int)
    two_electron_energy = df_noref.two_electron_energy.values.reshape(n, -1).astype(float)
    unrestricted = df_reaction.unrestricted.values.astype(bool)
    dataset = df_name.dataset.values

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
                'spin_one_hot': spin_one_hot,
                'dataset': dataset
                }

    return name_lookup

def get_portfolio_details(x, names):

    # Get the order by contribution of the portfolio
    idx = np.argsort(abs(x))

    w = []
    n = []

    for i in idx:
        weight = x[i]
        if abs(weight) < 1e-9:
            continue
        w.append(weight)
        n.append(names[i])

    return w, n

def outer_cv(x, y, m, params, outer_cv_splits, inner_cv_splits, grid = True):
    """
    Do outer cross validation to get the prediction errors of a method.
    """

    best_cv_params = []
    cv_portfolios = []
    errors = np.zeros(x.shape[0])

    if grid:
        cv_model = sklearn.model_selection.GridSearchCV
    else:
        cv_model = sklearn.model_selection.RandomizedSearchCV

    for i, (train_idx, test_idx) in enumerate(outer_cv_splits):
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx].reshape(1,-1), y[test_idx] # Remove reshape if not using leave one out
        if len(params) > 0:
            cvmod = cv_model(m, param_grid = params, scoring = 'neg_mean_absolute_error', iid=True,
                    return_train_score = False, cv = sklearn.model_selection.PredefinedSplit(inner_cv_splits[i]))
            cvmod.fit(train_x, train_y)
            cv_portfolios.append(cvmod.best_estimator_.portfolio)
            best_cv_params.append(cvmod.best_params_)
            y_pred = cvmod.predict(test_x)[0] # Remove [0] if not using leave one out
        else:
            m.fit(train_x, train_y)
            cv_portfolios.append(m.portfolio)
            best_cv_params.append(params)
            y_pred = m.predict(test_x)[0] # Remove [0] if not using leave one out

        errors[test_idx] = test_y - y_pred

    cv_portfolios = np.asarray(cv_portfolios)

    return errors, cv_portfolios, best_cv_params

def get_best_params(params):
    """
    Attempts to get the best set of parameters
    from a list of dictionaries containing the
    parameters that resulted in the best cv score
    """

    # Preprocess a bit
    d = {}
    for param in params:
        for key,value in param.items():
            if key not in d: d[key] = []
            d[key].append(value)

    # Select the most likely or median value
    for key,value in d.items():
        # if list choose most common
        if isinstance(value[0], str):
            best_value = max(set(value), key=value.count)
            d[key] = best_value
            continue

        # if numeric return median
        d[key] = np.median(value)

    return d


def leave_one_out_cv(data, include_other_reaction_types=False,
        include_other_spin_states=True, include_other_charge_states=True,
        include_other_datasets=True):
    """
    Attempts to do create leave one out cv splits,
    where none of the molecules in the test reaction
    can be present in the reactions of the training set.
    """
    def share_names(sub_names1, sub_names2, same_reaction_set):
        """
        Check if a name is present in both subsets. Special care needed
        for names with TS in them, since e.g. two molecules named TS1
        might be different structures.
        """
        for name1 in sub_names1:
            if name1 == "H":
                continue
            for name2 in sub_names2:
                if name2 == "H":
                    continue
                if name1 == name2:
                    if 'TS' in name1 and not same_reaction_set:
                        pass
                    else:
                        return True
        return False


    n = len(data['reaction_name'])

    splits = np.ones((n,n), dtype=bool) ^ np.diag(np.ones(n, dtype=bool))

    for i, name1 in enumerate(data['reaction_name']):
        # Get molecule names
        sub_names1 = re.split('\+|->', name1)
        rtype1 = data['reaction_class'][i]
        spin1 = data['spin'][i]
        charge1 = data['charge'][i]
        dataset1 = data['dataset'][i]
        for j, name2 in enumerate(data['reaction_name'][i+1:]):
            k = i+j+1
            sub_names2 = re.split('\+|->', name2)
            rtype2 = data['reaction_class'][k]
            if not include_other_reaction_types and rtype1 != rtype2:
                splits[i,k] = False
                splits[k,i] = False
                continue
            spin2 = data['spin'][k]
            if not include_other_spin_states and spin1 != spin2:
                splits[i,k] = False
                splits[k,i] = False
                continue
            charge2 = data['charge'][k]
            if not include_other_charge_states and charge1 != charge2:
                splits[i,k] = False
                splits[k,i] = False
                continue
            dataset2 = data['dataset'][k]
            if not include_other_datasets and dataset1 != dataset2:
                splits[i,k] = False
                splits[k,i] = False
                continue
            flag = share_names(sub_names1, sub_names2, (dataset1 == dataset2))
            splits[i,k] = (not flag)
            splits[k,i] = (not flag)

    outer_splits = [(np.where(splits[i])[0], i) for i in range(n)]

    inner_splits = []

    for i in range(n):
        indices = outer_splits[i][0]
        class1 = data['reaction_class'][i]
        classes = data['reaction_class'][indices]
        this_split = np.zeros(classes.size, dtype=int)
        this_split[np.where(classes != class1)[0]] = -1
        same_class_indices = np.where(classes == class1)[0]
        test_splits = [test for train,test in 
                sklearn.model_selection.KFold(3, shuffle=True).split(same_class_indices)]
        for j in range(1,3):
            this_split[same_class_indices[test_splits[j]]] = j

        inner_splits.append(this_split)

    return outer_splits, inner_splits

def get_best_params_and_model(x, y, model, best_cv_params, names):
    # Get the best params
    best_params = get_best_params(best_cv_params)

    # retrain the model on the full data
    model.set_params(**best_params)
    model.fit(x, y)
    portfolio = model.portfolio

    # Get human readable details on the selected portfolio
    portfolio_weight, portfolio_name = get_portfolio_details(portfolio, names)

    return best_params, portfolio, portfolio_weight, portfolio_name


if __name__ == "__main__":
    df = pd.read_pickle("pickles/combined_reac.pkl")
    data = parse_reaction_dataframe(df)
    #run_SingleMethod(data)
    run_LinearModel(data)
    quit()
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
