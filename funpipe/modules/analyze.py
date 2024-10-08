import numpy as np
from funfolding import solution as slt
import emcee
from iminuit import Minuit


def bin_test_data(f_test, g_test, target_bins, tree_binning_uniform, model):
    """
    Bins the measured data g_test based on the TreeBinning and the target f_test based on the target bins.
    Set f_test to None if unknown, return None for vec_f_test in this case.

    Parameters:
    ----------
    f_test : numpy.ndarray
        Target data. Set to None if unknown.
    g_test : numpy.ndarray
        Measured data.
    target_bins : numpy.ndarray
        Target bins.
    tree_binning_uniform : TreeBinningSklearn (Funfolding)
        Trained DecisionTree object to bin the observable space.
    model : LinearModel (Funfolding)
        Linear model f = A*g trained on the training data.

    Returns:
    -------
    vec_g_test : numpy.ndarray
        Vectorized g_test. 
    vec_f_test : numpy.ndarray
        Vectorized f_test. Return None if f_test is None.  
    """
    binned_g_test = tree_binning_uniform.digitize(g_test)

    if f_test is None:
        vec_g_test, vec_f_test = model.generate_vectors(
            digitized_obs=binned_g_test,
        )

    else:
        binned_f_test = np.digitize(f_test, target_bins)
        vec_g_test, vec_f_test = model.generate_vectors(
            digitized_obs=binned_g_test,
            digitized_truth=binned_f_test
        )
    return vec_g_test, vec_f_test

def unfold_data_mcmc(vec_g_test, model, n_used_steps, n_burn_steps, n_walkers, tau=None, reg_factor_f=1, error_calc='bayesian', move_object=emcee.moves.StretchMove()):
    """
    Unfolding the data by minimizing the likelihood function using MCMC.

    Parameters:
    ----------
    vec_g_test : numpy.ndarray
        Vectorized proxy data.
    model : LinearModel (Funfolding)
        Linear model f = A*g trained on the training data.
    n_used_steps : int
        Number of MCMC steps to be saved.
    n_burn_steps : int
        Number of burn-in steps.
    n_walkers : int
        Number of MCMC walkers.
    tau : float, optional
        Regularization strength but scales with 1/tau.
    reg_factor_f : vector, optional
        Regularization factor for each bin in the target space.

    Returns:
    -------
    f_est_mcmc : numpy.ndarray
        Estimated target data.
    lower_err : numpy.ndarray
        Lower error bound.
    upper_err : numpy.ndarray
        Upper error bound.
    """
    # Initialize likelihood 
    llh = slt.StandardLLH(
        tau=tau,
        C='thikonov',
        log_f=True,
        log_f_offset=1e-10,
        reg_factor_f=reg_factor_f #1/true_target
    )
    llh.initialize(
        vec_g=vec_g_test,
        model=model,
        ignore_n_bins_low=1,
        ignore_n_bins_high=1
    )

    # Employ MCMC walkers
    llh_mcmc = slt.LLHSolutionMCMC(
        n_used_steps=n_used_steps,
        n_walkers=n_walkers,
        n_burn_steps=n_burn_steps,
        random_state=42,
        error_calc=error_calc,
        move_object=move_object
    )
    llh_mcmc.initialize(model=llh.model, llh=llh)
    llh_mcmc.set_x0_and_bounds() # E>0 as bound possible?

    # Unfold
    f_est_mcmc, std_mcmc, sample_mcmc, probs_mcmc, autocorr_time_mcmc = llh_mcmc.fit()
    lower_err = f_est_mcmc - std_mcmc[0]
    upper_err = std_mcmc[1] - f_est_mcmc

    return f_est_mcmc, lower_err, upper_err, sample_mcmc, autocorr_time_mcmc

def unfold_data_minuit(vec_g_test, model, tau=None, reg_factor_f=1, simplex=True, ncall=None, errordef=Minuit.LIKELIHOOD, fixed=[]):
    """
    Unfolding the data by minimizing the likelihood function using Minuit.

    Parameters:
    ----------
    vec_g_test : numpy.ndarray
        Vectorized proxy data.
    model : LinearModel (Funfolding)
        Linear model f = A*g trained on the training data.
    tau : float, optional
        Regularization strength but scales with 1/tau.
    reg_factor_f : vector, optional
        Regularization factor for each bin in the target space.

    Returns:
    -------
    f_est_mcmc : numpy.ndarray
        Estimated target data.
    lower_err : numpy.ndarray
        Lower error bound.
    upper_err : numpy.ndarray
        Upper error bound.
    """
    # Initialize likelihood 
    llh = slt.StandardLLH(
        tau=tau,
        C='thikonov',
        log_f=True,
        log_f_offset=1e-10,
        reg_factor_f=reg_factor_f #1/true_target
    )
    llh.initialize(
        vec_g=vec_g_test,
        model=model,
        ignore_n_bins_low=1,
        ignore_n_bins_high=1
    )

    # Employ MCMC walkers
    llh_minuit = slt.LLHSolutionMinuit(
    )
    llh_minuit.initialize(model=llh.model, llh=llh)
    llh_minuit.set_x0_and_bounds() # E>0 as bound possible?

    # Unfold
    obj_minuit = llh_minuit.fit(ncall=ncall, simplex=simplex, errordef=errordef, fixed=fixed)
    obj_minuit.minos()

    return obj_minuit