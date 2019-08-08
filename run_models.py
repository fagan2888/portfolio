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
from sklearn.linear_model import Lasso, Ridge
import time

# Add the module to source path
dirname = os.path.dirname(os.path.realpath(__file__))
source_path = dirname + "/portfolio"
sys.path.append(source_path)

from portfolio.model import SingleMethod, LinearModel, Markowitz
import scipy.stats as ss

#def calc_mean_and_var(x):
#    vars_ = np.var(x, ddof = 1, axis = 1)
#    means = np.mean(x, axis = 1)
#    alphas = vars_ / sum(vars_)
#    mean = sum(means * alphas)

def check_for_errors(data):
    """
    Simple check for big errors in the data
    """
    dy = data['energy']-data['reference_energy'].reshape(-1,1)
    for i in range(data['reaction_name'].size):
        func_outliers = []
        for func in np.unique(data['functional']):
            idx = np.where(data['functional'] == func)[0]
            energy_func = dy[i, idx]
            median = np.median(energy_func)
            ## Find outliers
            outliers = np.where(abs(energy_func - median) > 100)[0]
            if len(outliers) == 0:
                continue
            func_outliers.extend(list(data['method_name'][idx][outliers]))
        basis_outliers = []
        for basis in np.unique(data['basis']):
            idx = np.where(data['basis'] == basis)[0]
            energy_func = dy[i, idx]
            median = np.median(energy_func)
            ## Find outliers
            outliers = np.where(abs(energy_func - median) > 100)[0]
            if len(outliers) == 0:
                continue
            basis_outliers.extend(list(data['method_name'][idx][outliers]))
        joint_outliers = list(set(basis_outliers) & set(func_outliers))
        if len(joint_outliers) == 0:
            continue
        print(data['reaction_name'][i], data['dataset'][i], joint_outliers)
        func = []
        basis = []
        for name in joint_outliers:
            if 'u-' in name:
                name = name[2:]
            f, b = name.split("/")
            func.append(f)
            basis.append(b)
        print(set(func), set(basis))

def run_SingleMethod(data, name):
    model = SingleMethod()
    cv_params = {'loss': ('mae', 'rmsd', 'max')}
    outer_splits, inner_splits = less_strict_leave_one_out_cv(data, include_other_reaction_types=True)

    run_method(data, model, cv_params, "pickles/%s_single_result.pkl" % name, outer_splits, inner_splits)

def run_LinearModel(data, name):
    model = LinearModel(positive_constraint=False)
    cv_params = {'l1_reg': 10**np.linspace(-1, 3, 40)}
    outer_splits, inner_splits = less_strict_leave_one_out_cv(data, include_other_reaction_types=True)
    run_method(data, model, cv_params, "pickles/%s_linear_result.pkl" % name, outer_splits, inner_splits)
    model = LinearModel(positive_constraint=True)
    cv_params = {}
    run_method(data, model, cv_params, "pickles/%s_linear_positive_result.pkl" % name, outer_splits, inner_splits)

def run_Markowitz(data, name):
    model = Markowitz(positive_constraint=False)
    cv_params = {'l1_reg': 10**np.linspace(-2, 2, 40)}
    outer_splits, inner_splits = less_strict_leave_one_out_cv(data, include_other_reaction_types=True)
    run_method(data, model, cv_params, "pickles/%s_markowitz_result.pkl" % name, outer_splits, inner_splits)
    model = Markowitz(positive_constraint=True)
    cv_params = {}
    run_method(data, model, cv_params, "pickles/%s_markowitz_positive_result.pkl" % name, outer_splits, inner_splits)


def run_method(data, model, cv_params, path, outer_splits, inner_splits):
    """
    Run all crossvalidation etc. needed to get results for
    a method. Store everything in a pickle.
    """
    if os.path.isfile(path):
        print("%s already generated" % path)
        return

    # create an empty file to parallelize tasks a bit easier
    with open(path, 'wb') as f:
        pass

    # Use only gga
    ggas = ['B88X', 'B', 'BECKE', 'B-LYP', 'B-P', 'B-VWN', 'CS', 'D', 'HFB', 'HFS',
            'LDA', 'LSDAC', 'LSDC', 'LYP88', 'PBE', 'PBEREV', 'PW91', 'S', 'SLATER', 'SOGGA11',
            'SOGGA', 'S-VWN', 'VS99', 'VWN80', 'VWN', 'M06-L','M11-L','MM06-L']

    gga_mask = np.isin(data['functional'], ggas)
    basis_mask0 = (data['basis'] == 'sto-3g')
    basis_mask1 = (data['basis'] == 'SV-P') | basis_mask0
    basis_mask2 = np.isin(data['basis'], ['svp','6-31+G-d,p']) | basis_mask1
    basis_mask3 = (data['basis'] == 'avdz') | basis_mask2
    basis_mask4 = (data['basis'] == 'tzvp') | basis_mask3
    basis_mask5 = (data['basis'] == 'avtz') | basis_mask4
    basis_mask6 = (data['basis'] == 'qzvp') | basis_mask5
    mp2_mask = (data['functional'] == 'df-lrmp2')
    wf_mask = mp2_mask | (data['functional'] == 'DCSD')
    unrestricted_mask = (data['unrestricted'] == True)

    subsets = [
            np.where(gga_mask & basis_mask1 & ~wf_mask)[0],
            np.where(gga_mask & basis_mask2 & ~wf_mask)[0],
            np.where(gga_mask & basis_mask3 & ~wf_mask)[0],
            np.where(gga_mask & basis_mask4 & ~wf_mask)[0],
            np.where(gga_mask & basis_mask5 & ~wf_mask)[0],
            np.where(gga_mask & basis_mask6 & ~wf_mask)[0],
            np.where(~gga_mask & basis_mask1 & ~wf_mask)[0],
            np.where(~gga_mask & basis_mask2 & ~wf_mask)[0],
            np.where(~gga_mask & basis_mask3 & ~wf_mask)[0],
            np.where(~gga_mask & basis_mask4 & ~wf_mask)[0],
            np.where(~gga_mask & basis_mask5 & ~wf_mask)[0],
            np.where(~gga_mask & basis_mask6 & ~wf_mask)[0]]

    for i, mask_i in enumerate([basis_mask1, basis_mask2, basis_mask3, basis_mask4, basis_mask5, basis_mask6]):
        for j, mask_j in enumerate([basis_mask1, basis_mask2, basis_mask3, basis_mask4, basis_mask5, basis_mask6]):
            if j > i:
                continue
            subsets.append(np.where((gga_mask & mask_i & ~wf_mask) | (~gga_mask & mask_j & ~wf_mask))[0])

    for i, mask_i in enumerate([basis_mask2, basis_mask3, basis_mask4, basis_mask5, basis_mask6]):
        for j, mask_j in enumerate([basis_mask2, basis_mask3, basis_mask4, basis_mask5, basis_mask6]):
            if j > i:
                continue
            subsets.append(np.where((gga_mask & mask_i & ~wf_mask) | (mask_j & mp2_mask))[0])
    for i, mask_i in enumerate([basis_mask2, basis_mask3, basis_mask4, basis_mask5, basis_mask6]):
        for j, mask_j in enumerate([basis_mask2, basis_mask3, basis_mask4, basis_mask5, basis_mask6]):
            if j > i:
                continue
            subsets.append(np.where((~gga_mask & mask_i & ~wf_mask) | (mask_j & mp2_mask))[0])



    subset_names = []
    for i in range(6):
        subset_names.append("gga%d" % (i+1))
    for i in range(6):
        subset_names.append("hybrid%d" % (i+1))

    for i, I in enumerate([1,2,3,4,5,6]):
        for j, J in enumerate([1,2,3,4,5,6]):
            if j > i:
                continue
            subset_names.append("gga%d+hybrid%d" % (I,J))

    for i, I in enumerate([2,3,4,5,6]):
        for j, J in enumerate([2,3,4,5,6]):
            if j > i:
                continue
            subset_names.append("gga%d+mp2%d" % (I,J))
    for i, I in enumerate([2,3,4,5,6]):
        for j, J in enumerate([2,3,4,5,6]):
            if j > i:
                continue
            subset_names.append("hybrid%d+mp2%d" % (I,J))

    assert(len(subset_names) == len(subsets))


    x = data['energy']
    y = data['reference_energy']

    d = {}

    errors = []
    cv_portfolios = []
    all_cv_params = []

    #dy = data['energy']-data['reference_energy'].reshape(-1,1)
    #idx = np.where(abs(dy) > 400)
    #for reaction, method in zip(*idx):
    #    print(data['dataset'][reaction], data['reaction_name'][reaction], data['method_name'][method], dy[reaction,method])
    #idx = np.where(abs(dy) > 200)
    #for reaction, method in zip(*idx):
    #    if 'sto-3g' in data['method_name'][method]:
    #        continue
    #    print(data['dataset'][reaction], data['reaction_name'][reaction], data['method_name'][method], dy[reaction,method])

    import time
    t = time.time()
    for i, subset in enumerate(subsets):
        print("subset", i, end='; ', flush=True)
        x_subset = x[:,subset].copy()
        y_subset = y.copy()
        # flip reactions if needed
        dy = x_subset - y_subset[:,None]
        dy_flipped = flip_order(dy)
        flip_idx = np.where(dy / dy_flipped < 0)[0]
        x_subset[flip_idx] *= -1
        y_subset[flip_idx] *= -1
        # Get best hyperparams from cv
        error, cv_portfolio, best_cv_params = \
                outer_cv(x_subset, y_subset, model, cv_params, outer_splits, inner_splits)

        # Convert the portfolios to the full set
        portfolio = np.zeros(x.shape, dtype=float)
        for i in range(x.shape[0]):
            portfolio[i,subset] = cv_portfolio[i]

        # Store various attributes for the given cost
        errors.append(error)
        cv_portfolios.append(portfolio)
        all_cv_params.append(best_cv_params)
        print("time", time.time()-t, flush=True)

    d['errors'] = np.asarray(errors)
    d['cv_portfolios'] = np.asarray(cv_portfolios)
    d['cv_params'] = all_cv_params
    d['reaction_names'] = data['reaction_name']
    d['method_names'] = data['method_name']
    d['dataset'] = data['dataset']
    d['reaction_class'] = data['reaction_class']
    d['subset_names'] = subset_names

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
    reaction_class = df_name.reaction_class.values.astype(int)
    ## Merge heavy atom and hydrogen into same classes, meaning
    ## class 5 is merged into 2 and 6 into 4
    #reaction_class[np.where(reaction_class == 5)[0]] = 2
    #reaction_class[np.where(reaction_class == 6)[0]] = 4
    ## Additionally merge all reactions and all barriers
    #reaction_class[np.where(reaction_class == 2)[0]] = 1
    #reaction_class[np.where(reaction_class == 4)[0]] = 3

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
    Could probably be replaced with cross_val_score or similar.
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
                    return_train_score = False, cv = inner_cv_splits[i],
                    n_jobs=1)
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

def less_strict_leave_one_out_cv(data, include_other_reaction_types=False,
        include_other_spin_states=True, include_other_charge_states=True,
        include_other_datasets=True, do_inner_splits=True):
    """
    Attempts to do create leave one out cv splits,
    where no reactions in the test set share the same left or 
    right hand side with the reactions of the training set.
    """
    def share_names(sub_names1, sub_names2, same_reaction_set):
        """
        Check if the same TS is present in both subsets.
        """
        for name1 in sub_names1:
            mol1 = set(name1.split("+"))
            for name2 in sub_names2:
                if len(mol1 | set(name2.split("+"))) == len(mol1) \
                    and len(mol1 & set(name2.split("+"))) == len(mol1):
                    if "TS" in name2 and not same_reaction_set:
                        continue
                    else:
                        return True
        return False


    n = len(data['reaction_name'])

    splits = np.ones((n,n), dtype=bool) ^ np.diag(np.ones(n, dtype=bool))

    for i, name1 in enumerate(data['reaction_name']):
        # Get molecule names
        sub_names1 = re.split('->', name1)
        rtype1 = data['reaction_class'][i]
        spin1 = data['spin'][i]
        charge1 = data['charge'][i]
        dataset1 = data['dataset'][i]
        for j, name2 in enumerate(data['reaction_name'][i+1:]):
            k = i+j+1
            sub_names2 = re.split('->', name2)
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

    if not do_inner_splits:
        return outer_splits

    inner_splits = []

    for i in range(n):
        original_indices = outer_splits[i][0]
        class1 = data['reaction_class'][i]
        classes = data['reaction_class'][original_indices]
        same_class_original_indices = original_indices[np.where(classes == class1)[0]]
        splits_i = []
        for j in same_class_original_indices:
            j_test_index = np.where(original_indices == j)[0][0]
            training = []
            for k in set(original_indices) & set(outer_splits[j][0]):
                k_test_index = np.where(original_indices == k)[0][0]
                training.append(k_test_index)
            splits_i.append((training, [j_test_index]))
        #inner_splits.append(((train, test) for train,test in splits_i))
        inner_splits.append(splits_i)


    return outer_splits, inner_splits

def leave_one_out_cv(data, include_other_reaction_types=False,
        include_other_spin_states=True, include_other_charge_states=True,
        include_other_datasets=True, do_inner_splits=True):
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

    if do_inner_splits == False:
        return outer_splits

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

def get_hydrogen_transfer_portfolios(data):
    # Get portfolio for hydrogen transfer enzyme system
    y = data['reference_energy']
    x = data['energy']
    methods = data['method_name']


    for i in range(2): # h-only or not
        if i == 0:
            idx_i = np.where((data['reaction_class'] == 5) | (data['reaction_class'] == 6))[0]
        else:
            idx_i = np.where((data['reaction_class'] == 5) | (data['reaction_class'] == 6) | (data['reaction_class'] == 2) | (data['reaction_class'] == 4))[0]

        for j in range(5):
            if j == 0: # sv(p)
               idx_j = np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0]
            if j == 1: # svp
               idx_j = np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0]
            elif j == 2: # avdz
               idx_j = np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p') | (data['basis'] == 'avdz')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0]
            elif j == 3: # no sto-3g
               idx_j = np.where(((data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p') | (data['basis'] == 'avdz')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0]
            elif j == 4: # lrmp2 sv(p)
               idx_j = np.where((((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | 
                   (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p') | (data['basis'] == 'avdz')) 
                   & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD')) | 
                   ((data['functional'] != 'df-lrmp2') & (data['basis'] == 'SV-P')))[0]
            elif j == 5: # lrmp2 svp
               idx_j = np.where((((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | 
                   (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p') | (data['basis'] == 'avdz')) 
                   & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD')) | 
                   ((data['functional'] != 'df-lrmp2') & ((data['basis'] == 'SV-P') | 
                       (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p'))))[0]

            outer_splits = leave_one_out_cv(data, include_other_reaction_types=True, do_inner_splits=False)

            for loss in ['mae', 'rmsd', 'max']:
                m = SingleMethod(loss=loss)
                errors = []
                count = []
                # Keeping track of the indices is a mess, so don't change
                for k, (train, test) in enumerate(outer_splits):
                    if k not in idx_i:
                        continue
                    actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
                    x_train = x[np.ix_(actual_train_idx, idx_j)]
                    x_test = x[test,idx_j].reshape(1,-1)
                    y_train = y[actual_train_idx]
                    y_test = y[test]
                    m.fit(x_train, y_train)
                    y_pred = m.predict(x_test)[0]
                    errors.append(y_pred - y_test)
                print(i, j, "single", loss, np.mean(np.abs(errors)))

            for l1 in list(10**np.linspace(0, 4, 50)):
                m = LinearModel(positive_constraint=False, l1_reg=l1, clip_value=1e-4)
                errors = []
                count = []
                # Keeping track of the indices is a mess, so don't change
                for k, (train, test) in enumerate(outer_splits):
                    if k not in idx_i:
                        continue
                    actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
                    x_train = x[np.ix_(actual_train_idx, idx_j)]
                    x_test = x[test,idx_j].reshape(1,-1)
                    y_train = y[actual_train_idx]
                    y_test = y[test]
                    m.fit(x_train, y_train)
                    y_pred = m.predict(x_test)[0]
                    errors.append(y_pred - y_test)
                print(i, j, "linear", l1, np.mean(np.abs(errors)))

            m = LinearModel(positive_constraint=True, l1_reg=0, clip_value=1e-4)
            errors = []
            count = []
            # Keeping track of the indices is a mess, so don't change
            for k, (train, test) in enumerate(outer_splits):
                if k not in idx_i:
                    continue
                actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
                x_train = x[np.ix_(actual_train_idx, idx_j)]
                x_test = x[test,idx_j].reshape(1,-1)
                y_train = y[actual_train_idx]
                y_test = y[test]
                m.fit(x_train, y_train)
                y_pred = m.predict(x_test)[0]
                errors.append(y_pred - y_test)
            print(i, j, "linear", "pos", np.mean(np.abs(errors)))

            for l1 in list(10**np.linspace(0, 4, 50)):
                m = Markowitz(positive_constraint=False, l1_reg=l1, clip_value=1e-4)
                errors = []
                count = []
                # Keeping track of the indices is a mess, so don't change
                for k, (train, test) in enumerate(outer_splits):
                    if k not in idx_i:
                        continue
                    actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
                    x_train = x[np.ix_(actual_train_idx, idx_j)]
                    x_test = x[test,idx_j].reshape(1,-1)
                    y_train = y[actual_train_idx]
                    y_test = y[test]
                    m.fit(x_train, y_train)
                    y_pred = m.predict(x_test)[0]
                    errors.append(y_pred - y_test)
                print(i, j, "markowitz", l1, np.mean(np.abs(errors)))

            m = Markowitz(positive_constraint=True, l1_reg=0, clip_value=1e-4)
            errors = []
            count = []
            # Keeping track of the indices is a mess, so don't change
            for k, (train, test) in enumerate(outer_splits):
                if k not in idx_i:
                    continue
                actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
                x_train = x[np.ix_(actual_train_idx, idx_j)]
                x_test = x[test,idx_j].reshape(1,-1)
                y_train = y[actual_train_idx]
                y_test = y[test]
                m.fit(x_train, y_train)
                y_pred = m.predict(x_test)[0]
                errors.append(y_pred - y_test)
            print(i, j, "markowitz", "pos", np.mean(np.abs(errors)))

# TODO make inner splits in a more stratified way
def better_leave_one_out_cv(data, include_other_reaction_types=False,
        include_other_spin_states=True, include_other_charge_states=True,
        include_other_datasets=True, do_inner_splits=True, strict=True):
    """
    Attempts to create leave one out cv splits where heavily correlated
    reactions are removed from the training and validation sets.
    """

    n = len(data['reaction_name'])

    splits = np.ones((n,n), dtype=bool) ^ np.diag(np.ones(n, dtype=bool))

    for i, name1 in enumerate(data['reaction_name']):
        # Get molecule names
        rtype1 = data['reaction_class'][i]
        spin1 = data['spin'][i]
        charge1 = data['charge'][i]
        dataset1 = data['dataset'][i]
        for j, name2 in enumerate(data['reaction_name'][i+1:]):
            k = i+j+1
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

    # Get the automated list of what to leave out
    idx = find_correlations(data, strict)
    for i in range(n):
        splits[i, idx[i]] = False

    outer_splits = [(np.where(splits[i])[0], i) for i in range(n)]

    if not do_inner_splits:
        return outer_splits

    inner_splits = []

    for i in range(n):
        indices = outer_splits[i][0]
        class1 = data['reaction_class'][i]
        classes = data['reaction_class'][indices]
        this_split = np.zeros(classes.size, dtype=int)
        this_split[np.where(classes != class1)[0]] = -1
        same_class_indices = np.where(classes == class1)[0]
        test_splits = [test for train,test in 
                sklearn.model_selection.KFold(5, shuffle=True).split(same_class_indices)]
        for j in range(1,5):
            this_split[same_class_indices[test_splits[j]]] = j

        #print(test_splits[0])
        #subset = set(outer_splits[outer_splits[i][0][test_splits[0][0]]][0])
        #for j in test_splits[0][1:]:
        #    subset = subset & set(outer_splits[outer_splits[i][0][j]][0])
        #print(outer_splits[i][0])
        #print(subset)

        inner_splits.append(this_split)

    return outer_splits, inner_splits


def find_correlations(data, strict=True):
    """
    Do a bunch of fancy stuff to find correlated reactions
    """

    np.random.seed(42)

    output1 = []

    # First check covariance between the different reactions, using
    # all methods that has a maximum error of less than 15 kcal/mol.
    idx = np.where((abs(data['energy']-data['reference_energy'][:,None]) < 15).all(0))[0]
    X = data['energy'][:,idx] - data['reference_energy'][:,None]
    cov = np.cov(X, ddof = 1, rowvar = True)
    diag = np.diag(cov)
    cov /= np.sqrt(diag)[:,None] * np.sqrt(diag)[None,:]
    if strict:
        upper_limit = 0.9
    else:
        upper_limit = 0.95
    for i in range(102):
        idx = np.where(np.abs(cov[i]) > upper_limit)[0].tolist()
        for j, J in enumerate(idx):
            if i == J:
                idx.pop(j)

        output1.append(idx)

    #l = []
    #for i in range(102):
    #    for j in range(i+1, 102):
    #        l.append(abs(cov[i,j]))
    #print(np.percentile(l, 95))
    #quit()


    # Get linear network
    reactions = []
    for i in range(102):
        reaction_name = data['reaction_name'][i]
        reactants, products = [name.split("+") for name in reaction_name.split("->")]
        for j, reactant in enumerate(reactants):
            name = data['dataset'][i] + "_" + reactant
            # Lots of similar or same molecules across the sets
            if name in ["htbh38_C2H6","abde12_C2H6"]:
                name = "C2H6"
            elif name in ["htbh38_CH3", "abde12_CH3", "nhtbh38_CH3"]:
                name = "CH3"
            elif name in ["htbh38_H", "abde12_H", "nhtbh38_H"]:
                name = "H"
            elif name in ["htbh38_OH", "abde12_OH", "nhtbh38_OH"]:
                name = "OH"
            elif name in ["nhtbh38_CH3CH2", "htbh38_C2H5"]:
                name = "C2H5"
            elif name in ["htbh38_Cl", "nhtbh38_Cl"]:
                name = "Cl"
            elif name in ["htbh38_F", "nhtbh38_F"]:
                name = "F"
            elif name in ["htbh38_HCl", "nhtbh38_HCl"]:
                name = "HCl"
            elif name in ["htbh38_HF", "nhtbh38_HF"]:
                name = "HF"

            reactants[j] = name

        for j, product in enumerate(products):
            name = data['dataset'][i] + "_" + product
            # Lots of similar or same molecules across the sets
            if name in ["htbh38_C2H6","abde12_C2H6"]:
                name = "C2H6"
            elif name in ["htbh38_CH3", "abde12_CH3", "nhtbh38_CH3"]:
                name = "CH3"
            elif name in ["htbh38_H", "abde12_H", "nhtbh38_H"]:
                name = "H"
            elif name in ["htbh38_OH", "abde12_OH", "nhtbh38_OH"]:
                name = "OH"
            elif name in ["nhtbh38_CH3CH2", "htbh38_C2H5"]:
                name = "C2H5"
            elif name in ["htbh38_Cl", "nhtbh38_Cl"]:
                name = "Cl"
            elif name in ["htbh38_F", "nhtbh38_F"]:
                name = "F"
            elif name in ["htbh38_HCl", "nhtbh38_HCl"]:
                name = "HCl"
            elif name in ["htbh38_HF", "nhtbh38_HF"]:
                name = "HF"

            products[j] = name

        reactions.append((copy.copy(reactants), copy.copy(products)))

    molecules = []
    for reactants, products in reactions:
        molecules.extend(reactants)
        molecules.extend(products)

    uniq_mols = np.unique(molecules)

    X = np.zeros((len(reactions), uniq_mols.size))
    for i in range(len(reactions)):
        reactants, products = reactions[i]
        for mol in reactants:
            idx = np.where(uniq_mols == mol)[0][0]
            X[i, idx] -= 1
        for mol in products:
            idx = np.where(uniq_mols == mol)[0][0]
            X[i, idx] += 1


    output2 = []
    # Try and see if any reaction can be described as a
    # linear combination of other reactions
    # Use both a lasso and ridge model with loose cutoffs
    # and take the joint set
    if strict:
        l1 = 1e-2
        cutoff = 0.1
    else:
        l1 = 1.7e-1#8e-2
        cutoff = 0.15
    m1 = Lasso(alpha=l1) # loose 1e-2
    m2 = Ridge(alpha=1e-6)
    energies = np.random.random((uniq_mols.size, 5000))
    Y = X.dot(energies)
    for i in range(102):
        indices = np.asarray(list(range(i)) + list(range(i+1,102)))
        m1.fit(Y[indices].T, Y[i])
        m2.fit(Y[indices].T, Y[i])
        idx1 = np.where(abs(m1.coef_) > 0)[0]
        idx2 = np.where(abs(m2.coef_) > cutoff)[0] # loose 0.1

        idx = list(set(idx1) & set(idx2))

        output2.append(indices[idx].tolist())

    # See if we can predict the reaction energy by knowing the reaction network.
    # This doesn't seem to be meaningful, since our fitted methods are not given
    # the actual reaction network
    cum_error = np.zeros(len(reactions))

    m = Lasso(alpha=1e-6)
    for j in range(20):
        # Make up some fake energies
        mol_ene = np.random.random(uniq_mols.size)
        y = np.sum(X * mol_ene[None,:], axis=1)

        for i in range(102):
            indices = list(range(i)) + list(range(i+1,102))
            m.fit(X[indices], y[indices])
            cum_error[i] += abs(m.predict(X[i:i+1])[0] - y[i])

    print(data['reaction_name'][np.where(cum_error < 20*1e-3)[0]])
    quit()

    return [sorted(list(set(output1[i]) | set(output2[i]))) for i in range(102)]

def test_method(data):
    y = data['reference_energy']
    x = data['energy']

    #data['reaction_class'][np.where(data['reaction_class'] == 5)[0]] = 2
    #data['reaction_class'][np.where(data['reaction_class'] == 6)[0]] = 4

    subsets = [
               np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0],
               np.where(((data['basis'] == 'sto-3g') | (data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0],
               np.where(((data['basis'] == 'SV-P')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0],
               np.where(((data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD'))[0],
               np.where((((data['basis'] == 'SV-P') | (data['basis'] == 'svp') | (data['basis'] == '6-31+G-d,p')) & (data['functional'] != 'df-lrmp2') & (data['functional'] != 'DCSD')) | ((data['functional'] == 'df-lrmp2') & (data['basis'] == 'svp')))[0],
              ]
    #for strict in True,False:
    #    for include in True,False:
    outer_splits = less_strict_leave_one_out_cv(data, include_other_reaction_types=False, do_inner_splits=False)
    for i in [6]:#range(1,5):
        idx_i = np.where(data['reaction_class'] == i)[0]
        for j, idx_j in enumerate(subsets):
            if j != 1:
                continue
            l1s = 10**np.linspace(-2, 2, 20)
            #mae = []
            #for l1 in l1s:
            #    m = LinearModel(positive_constraint=False, l1_reg=l1, integer_constraint=False, clip_value=1e-4)
            #    errors = []
            #    # Keeping track of the indices is a mess, so don't change
            #    for k, (train, test) in enumerate(outer_splits):
            #        if k not in idx_i:
            #            continue
            #        actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
            #        x_train = x[np.ix_(actual_train_idx, idx_j)]
            #        x_test = x[test,idx_j].reshape(1,-1)
            #        y_train = y[actual_train_idx]
            #        y_test = y[test]
            #        m.fit(x_train, y_train)
            #        y_pred = m.predict(x_test)[0]
            #        errors.append(y_pred - y_test)
            #    mae.append(np.mean(np.abs(errors)))
            #    #print(include, strict, "linear", np.log10(l1), np.mean(np.abs(errors)))
            #idx = np.argmin(mae)
            #print(strict, include, i, j, "linear", np.log10(l1s[idx]), mae[idx-2:idx+3])

            #for ub in (0.5, 0.25, 0.1, 0.05):
            #    mae = []
            #    for l1 in l1s:
            #        m = Markowitz(positive_constraint=False, l1_reg=l1, integer_constraint=False, clip_value=1e-4,
            #                method='mean_upper_bound_min_variance', upper_bound=ub)
            #        errors = []
            #        # Keeping track of the indices is a mess, so don't change
            #        for k, (train, test) in enumerate(outer_splits):
            #            if k not in idx_i:
            #                continue
            #            actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
            #            x_train = x[np.ix_(actual_train_idx, idx_j)]
            #            x_test = x[test,idx_j].reshape(1,-1)
            #            y_train = y[actual_train_idx]
            #            y_test = y[test]
            #            m.fit(x_train, y_train)
            #            y_pred = m.predict(x_test)[0]
            #            errors.append(y_pred - y_test)
            #        mae.append(np.mean(np.abs(errors)))
            #        #print(include, strict, "markowitz", np.log10(l1), np.mean(np.abs(errors)))
            #    idx = np.argmin(mae)
            #    print(strict, include, i, j, "markowitz2", ub, np.log10(l1s[idx]), mae[idx])
            m = Markowitz(positive_constraint=False, l1_reg=10**-0.5, integer_constraint=False, clip_value=0)
            errors = []
            # Keeping track of the indices is a mess, so don't change
            for k, (train, test) in enumerate(outer_splits):
                if k not in idx_i:
                    continue
                actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
                x_train = x[np.ix_(actual_train_idx, idx_j)]
                x_test = x[test,idx_j].reshape(1,-1)
                y_train = y[actual_train_idx]
                y_test = y[test]
                m.fit(x_train, y_train)
                y_pred = m.predict(x_test)[0]
                errors.append(y_pred - y_test)
            print(np.mean(np.abs(errors)))
            #print(include, strict, "markowitz", np.log10(l1), np.mean(np.abs(errors)))

            #mae = []
            #for l1 in l1s:
            #    m = Markowitz(positive_constraint=False, l1_reg=l1, integer_constraint=False, clip_value=0)
            #    errors = []
            #    # Keeping track of the indices is a mess, so don't change
            #    for k, (train, test) in enumerate(outer_splits):
            #        if k not in idx_i:
            #            continue
            #        actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
            #        x_train = x[np.ix_(actual_train_idx, idx_j)]
            #        x_test = x[test,idx_j].reshape(1,-1)
            #        y_train = y[actual_train_idx]
            #        y_test = y[test]
            #        m.fit(x_train, y_train)
            #        y_pred = m.predict(x_test)[0]
            #        errors.append(y_pred - y_test)
            #    mae.append(np.mean(np.abs(errors)))
            #    #print(include, strict, "markowitz", np.log10(l1), np.mean(np.abs(errors)))
            #idx = np.argmin(mae)
            #print(i, j, "markowitz", np.log10(l1s[idx]), mae[idx-2:idx+3])

            #mae = []
            #losses = ('max', 'mae', 'rmsd')
            #for loss in losses:
            #    m = SingleMethod(loss=loss)
            #    errors = []
            #    count = []
            #    # Keeping track of the indices is a mess, so don't change
            #    for k, (train, test) in enumerate(outer_splits):
            #        if k not in idx_i:
            #            continue
            #        actual_train_idx = np.asarray(list(set(idx_i) & set(train)))
            #        x_train = x[np.ix_(actual_train_idx, idx_j)]
            #        x_test = x[test,idx_j].reshape(1,-1)
            #        y_train = y[actual_train_idx]
            #        y_test = y[test]
            #        m.fit(x_train, y_train)
            #        y_pred = m.predict(x_test)[0]
            #        errors.append(y_pred - y_test)
            #    mae.append(np.mean(np.abs(errors)))
            #    #print(include, strict, "single", loss, np.mean(np.abs(errors)))
            #print(strict, include, i, j, "single", losses[np.argmin(mae)], min(mae))


def reaction_correlation(data):
    import seaborn as sns
    import scipy.stats as ss
    sortidx = np.argsort(data['reaction_class'])
    data['reaction_class'] = data['reaction_class'][sortidx]
    data['energy'] = data['energy'][sortidx]
    data['reference_energy'] = data['reference_energy'][sortidx]
    data['reaction_name'] = data['reaction_name'][sortidx]
    dy = data['energy'] - data['reference_energy'][:,None]
    dy = dy[:,np.where((dy < 15).all(0))[0]]
    #idx = np.where((data['reaction_class'] == 1) | (data['reaction_class'] == 2) | (data['reaction_class'] == 5))[0]
    idx = np.where((data['reaction_class'] == 2) | (data['reaction_class'] == 5))[0]
    #idx = np.where((data['reaction_class'] == 2))[0]
    #for i in range(1,7):
    #    x = np.where(data['reaction_class'] == i)[0]
    #    print(i, x.min(), x.max())
    #dy = dy[:,np.where((dy < 10).all(0))[0]]
    dy = dy[idx]
    dy = flip_order(dy)

    #cov = np.zeros((dy.shape[0],dy.shape[0]))
    #for i in range(dy.shape[0]):
    #    for j in range(i+1, dy.shape[0]):
    #        t,p = ss.wilcoxon(dy[i], dy[j])
    #        logp = np.log(p)
    #        cov[i,j] = logp
    #        cov[j,i] = logp

    cov = np.corrcoef(dy, ddof=1)

    # Plot heatmap
    sns.heatmap((cov), square=True, linewidths=.25, cbar_kws={"shrink": .5},
            cmap = sns.diverging_palette(220, 10, as_cmap=True),
            xticklabels=data['reaction_name'][idx], yticklabels=data['reaction_name'][idx],
            center=0, vmax=1, vmin=-1)
    plt.xticks(rotation=-90)
    plt.yticks(rotation=0)
    # Plot grid
    n = len(idx)
    for i in sorted(np.unique(data['reaction_class'][idx]))[1:]:
        m = np.where(data['reaction_class'][idx] == i)[0][0]
        plt.plot([0,n], [m,m], "-", c='k')
        plt.plot([m,m], [0,n], "-", c='k')
    plt.show()
    #sns.heatmap(abs(cov), square=True, linewidths=.25, cbar_kws={"shrink": .5},
    #        cmap = sns.diverging_palette(220, 10, as_cmap=True),
    #        xticklabels=data['reaction_name'][idx], yticklabels=data['reaction_name'][idx],
    #        center=0, vmax=1)
    #plt.xticks(rotation=-90)
    #plt.yticks(rotation=0)
    ## Plot grid
    #n = len(idx)
    #for i in range(1,6):
    #    m = np.where(data['reaction_class'] == i+1)[0][0]
    #    plt.plot([0,n], [m,m], "-", c='k')
    #    plt.plot([m,m], [0,n], "-", c='k')
    #plt.show()

def flip_order(x):
    """
    For e.g. reaction energies, the order of the reaction is arbitrary.
    This tries to flip the order of the reactions one by one and selects
    the order that maximises the likelihood.
    """
    def get_nll(x):
        """
        Returns the negative log-likelihood
        """

        # Get means of all methods
        means = x.mean(0)
        # Get the covariance
        cov = np.cov(x, ddof=1, rowvar=False)
        # Add some regularization to make the covariance non-singular
        cov += np.identity(cov.shape[0]) * 5e-4
        # Creates the multivariate normal
        rv = ss.multivariate_normal(means, cov)
        # Gets the negative log likelihood
        nll = -sum(rv.logpdf(x))
        return nll

    last_nll = get_nll(x)


    # Run through the data a maximum of 10 times.
    for j in range(10):
        # Keep track of whether a flip has occured in this iteration
        has_flipped = False
        # Loop through all data points
        for i in range(x.shape[0]):
            # Do the flip
            xflipi = x.copy()
            xflipi[i] *= -1
            # Recalculate the nll
            nll = get_nll(xflipi)
            if nll < last_nll:
                x[i] *= -1
                last_nll = nll
                has_flipped = True
        # Break if no flip
        if has_flipped == False:
            break
    return x

if __name__ == "__main__":
    df = pd.read_pickle("pickles/combined_reac.pkl")
    data = parse_reaction_dataframe(df)
    ## Merge heavy atom and hydrogen into same classes, meaning
    ## class 5 is merged into 2 and 6 into 4
    #data['reaction_class'][np.where(data['reaction_class'] == 5)[0]] = 2
    #data['reaction_class'][np.where(data['reaction_class'] == 6)[0]] = 4

    ### check for outliers
    ##check_for_errors(data)
    #run_SingleMethod(data)
    #run_LinearModel(data)
    #run_Markowitz(data)

    ### Additionally merge all reactions and all barriers
    #data['reaction_class'][np.where(data['reaction_class'] == 2)[0]] = 1
    #data['reaction_class'][np.where(data['reaction_class'] == 4)[0]] = 3

    #run_SingleMethod(data, "merged_single_method")
    #run_LinearModel(data, "merged_linear_method")
    #run_Markowitz(data, "merged_markowitz")

    #get_hydrogen_transfer_portfolios(data)
    #test_method(data)
    #reaction_correlation(data)
    #ocv, icv = less_strict_leave_one_out_cv(data, include_other_reaction_types=True, do_inner_splits=True)
    #for i in range(len(ocv)):
    #    all_ = list(range(len(ocv)))
    #    all_.pop(i)
    #    print(data['reaction_name'][ocv[i][0][np.where(icv[i] == 0)[0]]])
    #    quit()

    #    diff = np.setdiff1d(all_, cv[i][0])
    #    print(data['reaction_name'][i], data['reaction_name'][diff])

    # Loop through the subsets we want to study
    for i in range(8):
        print("class", i)
        if i == 0:
            idx = np.where(data['reaction_class'] == 1)[0]
        elif i == 1:
            idx = np.where(data['reaction_class'] == 4)[0]
        elif i == 2:
            idx = np.where(data['reaction_class'] == 5)[0]
        elif i == 3:
            idx = np.where(data['reaction_class'] == 6)[0]
        elif i == 4:
            idx = np.where((data['reaction_class'] == 2) & (data['reaction_class'] == 5))[0]
        elif i == 5:
            idx = np.where((data['reaction_class'] == 4) & (data['reaction_class'] == 6))[0]
        elif i == 6:
            idx = np.where((data['reaction_class'] == 3) & (data['reaction_class'] == 4) & (data['reaction_class'] == 6))[0]
        elif i == 7:
            idx = np.arange(0,data['reaction_class'].size)

        name = str(i + 1)

        subdata = {}
        for key in data.keys():
            if data[key].shape[0] == data['reaction_class'].shape[0]:
                subdata[key] = data[key][idx]
            else:
                subdata[key] = data[key]

        print("single")
        run_SingleMethod(data, name)
        print("linear")
        run_LinearModel(data, name)
        print("markowitz")
        run_Markowitz(data, name)

