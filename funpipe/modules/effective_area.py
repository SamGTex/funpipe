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

def calc_effective_area(df_lvl0, df_lvl2, varname_energy, varname_zenith, bins_energy, bins_zenith, varname_weights='weights', path_build=None, solid_angle=2*np.pi):

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