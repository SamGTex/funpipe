import numpy as np
from funfolding import solution as slt

def bin_test_data(f_test, g_test, target_bins, tree_binning_uniform, model):
    """
    Bins the test data based on the target bins and tree binning uniform.

    Parameters:
    - f_test (numpy.ndarray): Test data for f.
    - g_test (numpy.ndarray): Test data for g.
    - target_bins (numpy.ndarray): Target bins for digitizing f_test.
    - tree_binning_uniform: Tree binning uniform object for digitizing g_test.
    - model: Model object.

    Returns:
    - vec_g_test (numpy.ndarray): Vectorized g_test.
    - vec_f_test (numpy.ndarray): Vectorized f_test.
    """
    binned_g_test = tree_binning_uniform.digitize(g_test)
    binned_f_test = np.digitize(f_test, target_bins)
    vec_g_test, vec_f_test = model.generate_vectors(
        digitized_obs=binned_g_test,
        digitized_truth=binned_f_test
    )
    return vec_g_test, vec_f_test

def unfold_data(vec_g_test, model, n_used_steps, n_burn_steps, n_walkers, tau=None, reg_factor_f=1):
    """
    Unfolds the data using MCMC walkers.

    Parameters:
    - vec_g_test (numpy.ndarray): Vectorized g_test.
    - model: Model object.
    - n_used_steps (int): Number of used steps for MCMC walkers.
    - n_burn_steps (int): Number of burn steps for MCMC walkers.
    - n_walkers (int): Number of MCMC walkers.
    - tau (float): Tau value for the likelihood. Default is None.
    - reg_factor_f (float): Regularization factor for f. Default is 1.

    Returns:
    - f_est_mcmc (numpy.ndarray): Estimated f using MCMC walkers.
    - lower_err (numpy.ndarray): Lower error bound for f_est_mcmc.
    - upper_err (numpy.ndarray): Upper error bound for f_est_mcmc.
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
    )
    llh_mcmc.initialize(model=llh.model, llh=llh)
    llh_mcmc.set_x0_and_bounds() # E>0 as bound possible?

    # Unfold
    f_est_mcmc, std_mcmc, sample_mcmc, probs_mcmc, autocorr_time_mcmc = llh_mcmc.fit()
    lower_err = f_est_mcmc - std_mcmc[0]
    upper_err = std_mcmc[1] - f_est_mcmc

    return f_est_mcmc, lower_err, upper_err


