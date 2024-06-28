import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

def A_projected(zenith, r=600, l=1200):
    '''
    Calculate the projected area of the detector, approximated as a cylinder with radius r and length l.

    Parameters:
    -----------
    zenith : float or array-like
    r : float
        Radius of the cylinder.
    l : float
        Length of the cylinder.

    Returns:
    --------
    A : float or array-like
        Projected area of the cylinder.
    '''
    A = np.pi * r * r * np.cos(zenith) + 2 * r * l * np.sin(zenith)
    return A

def get_solid_angle(theta_max):
    '''
    Calculate the solid angle of a cone with opening angle theta_max.

    Parameters:
    -----------
    theta_max : float
        Opening angle of the cone in degrees.

    Returns:
    --------
    solid_angle : float
        Solid angle of the cone in steradians.
    '''
    
    return 2 * np.pi * (1 - np.cos(np.deg2rad(theta_max)))

def create_bins_const_solid_angle(theta_min, theta_max, N_bins):
    '''
    Create bins with a constant solid angle.

    Parameters:
    -----------
    theta_min : float
        Minimum zenith angle in degrees.
    theta_max : float
        Maximum zenith angle in degrees.
    N_bins : int
        Number of bins.

    Returns:
    --------
    bins : array-like
        Bins with constant solid angle.
    '''

    cos_min = np.cos(np.deg2rad(theta_min))
    cos_max = np.cos(np.deg2rad(theta_max))

    bins = np.arccos(np.linspace(cos_min, cos_max, N_bins + 1))
    bins = np.rad2deg(bins)

    return bins

def calc_effective_area(df_lvl0, df_lvl2, varname_energy, varname_zenith, bins_energy, bins_zenith, varname_weights='weights', path_build=None, solid_angle=2*np.pi):
    '''
    Calculate the effective area A and the correction factor A / (dE * dOmega) for a given set of level0 and level2 data.
    Using the formula A = A_proj * N_lvl2 / N_lvl0 for each energy and zenith bin.

    Parameters:
    -----------
    df_lvl0 : pd.DataFrame
        All generated events that would have hit the detector considering only the angle.

    df_lvl2 : pd.DataFrame
        Selected events in the analysis.

    varname_energy : str
        Name of the energy variable in the dataframes.

    varname_zenith : str
        Name of the zenith variable in the dataframes.

    bins_energy : array-like
        Energy bins.

    bins_zenith : array-like
        Zenith bins.

    varname_weights : str
        Name of the weights variable in the dataframes.

    path_build : str
        Path to save plots of level0 and level2 data binned.
        If None, no plots are saved.

    solid_angle : float
        Solid angle of the detector in steradians.
        Default is 2 * np.pi, but needs to be adjusted for zenith cuts.

    Returns:
    --------
    correction_factor : array-like
        Correction factor A / (dE * dOmega) for each energy bin.
        Can be used to correct the event rate to a relative flux.
    '''

    # hist level0 data
    plt.figure(figsize=(6, 4), dpi=300)
    level0_map, _, _, _ = plt.hist2d(
    df_lvl0[varname_energy],
    np.rad2deg(df_lvl0[varname_zenith]),
    bins=[bins_energy, bins_zenith],
    weights=df_lvl0[varname_weights],
    norm=mpl.colors.LogNorm(),
    )

    # plot if path_build is given
    if path_build is not None:
        plt.title('Level 0')
        plt.xscale('log')
        plt.xlabel('Energy at Surface / GeV')
        plt.ylabel(r'$\Theta$ / °')
        plt.colorbar(label = 'Event Rate / Hz')
        plt.savefig(os.path.join(path_build, 'hist2d_level0.png'))
        plt.close()

    # hist level2 data
    plt.figure(figsize=(6, 4), dpi=300)
    level2_map, _, _, _ = plt.hist2d(
    df_lvl2[varname_energy],
    np.rad2deg(df_lvl2[varname_zenith]),
    bins=[bins_energy, bins_zenith],
    weights=df_lvl2[varname_weights],
    norm=mpl.colors.LogNorm(),
    )

    # plot if path_build is given
    if path_build is not None:
        plt.title('Level 2')
        plt.xscale('log')
        plt.xlabel('Energy at Surface / GeV')
        plt.ylabel(r'$\Theta$ / °')
        plt.colorbar(label = 'Event Rate / Hz')
        plt.savefig(os.path.join(path_build, 'hist2d_level2.png'))
        plt.close()

    # projected area for each zenith bin
    zenith_bincenters = bins_zenith[:-1] + np.diff(bins_zenith)/2
    projected_areas = A_projected(np.deg2rad(zenith_bincenters))
    
    # calc 2D effective area
    A_eff_2d = projected_areas * level2_map / level0_map
    A_eff_2d[np.isfinite(A_eff_2d) == False] = 0

    # calc 1D effective area
    A_eff = np.average(A_eff_2d, axis=1, weights=level0_map)

    # correction factor
    correction_factor = A_eff * np.diff(bins_energy) * solid_angle

    return correction_factor

# Bootstrapping (lvl2 data)
def eff_area_bootstrapping(num_iter, df_lvl0, df_lvl2, varname_energy, varname_zenith, bins_energy, bins_zenith, varname_weights='weights', path_build=None, solid_angle=2*np.pi):
    '''
    Calculate the effective area A and returns the resulting correction factor A / (dE * dOmega) for a given set of level0 and level2 data using bootstrapping to estimate the uncertainty.
    The calculation of the effective area is based on the formula A = A_proj * N_lvl2 / N_lvl0 for each energy and zenith bin.

    Parameters:
    -----------
    num_iter : int
        Number of bootstrapping iterations.

    df_lvl0 : pd.DataFrame
        All generated events that would have hit the detector considering only the angle.

    df_lvl2 : pd.DataFrame
        Selected events in the analysis.

    varname_energy : str
        Name of the energy variable in the dataframes.

    varname_zenith : str
        Name of the zenith variable in the dataframes.

    bins_energy : array-like
        Energy bins.

    bins_zenith : array-like
        Zenith bins.

    varname_weights : str
        Name of the weights variable in the dataframes.

    path_build : str
        Path to save plots of level0 and level2 data binned.
        If None, no plots are saved.

    solid_angle : float
        Solid angle of the detector in steradians.
        Default is 2 * np.pi, but needs to be adjusted for zenith cuts.

    Returns:
    --------
    stat_median : array-like
        Median of the correction factor A / (dE * dOmega) for each energy bin.
        Can be used to correct the event rate to a relative flux.

    stat_quantiles : 2D array-like
        The 16th and 84th quantiles of the correction factor A / (dE * dOmega) for each energy bin.
        Can be used to estimate the uncertainty of the correction factor.
    '''
    
    stat_aeff = []

    # norm weights
    weights_normed_lvl2 = df_lvl2[varname_weights] / df_lvl2[varname_weights].sum()
    weights_normed_lvl0 = df_lvl0[varname_weights] / df_lvl0[varname_weights].sum()

    for i in range(num_iter):
        # Draw random indices accoring to normed weights with replacement
        np.random.seed(i)
        inds_lvl2 = np.random.choice(
            np.arange(len(weights_normed_lvl2)),
            p=weights_normed_lvl2,
            replace=True,
            size=len(weights_normed_lvl2),
        )

        #inds_lvl0 = np.random.choice(
        #    np.arange(len(weights_normed_lvl0)),
        #    p=weights_normed_lvl0,
        #    replace=True,
        #    size=len(weights_normed_lvl0),
        #)

        # Save all boostrapping iterations
        stat_aeff += [
            calc_effective_area(df_lvl0, df_lvl2.iloc[inds_lvl2], varname_energy, varname_zenith, bins_energy, bins_zenith, varname_weights='weights', path_build=path_build)
        ]

    stat_aeff = np.array(stat_aeff)

    # calc. mean and quantiles
    stat_median = np.median(stat_aeff, axis=0)
    stat_quantiles = np.abs(
        np.quantile(stat_aeff, [0.16, 0.84], axis=0) - stat_median
    )

    # save to csv
    if path_build is not None:
        _data = {
            "median": stat_median,
            "quantile_0.16": stat_quantiles[0],
            "quantiles_0.84": stat_quantiles[1],
        }
        _df = pd.DataFrame(_data)
        _df.to_csv(os.path.join(path_build, "correction_factor.csv"))

    return stat_median, stat_quantiles