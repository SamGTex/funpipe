import numpy as np
import matplotlib.pyplot as plt

# Import for script within the pipeline project
#try:
#    from utils.calc import ratio_error
#except ImportError:
#    pass

# Import for external script
#try:
#    from pipeline.utils.calc import ratio_error
#except ImportError:
#    pass

from .calc import ratio_error

def fill_between_axes(axes, bins, y1, y2, color, alpha=0.5):

    '''
    Fill the area between two lines in a histogram plot.

    Parameters:
    -----------
    axes : matplotlib.pyplot.axis
        Where to pot the data.

    bins : array-like
        The bin edges of the histogram.
    
    y1 : array-like
        The lower boundary of the area to fill.

    y2 : array-like
        The upper boundary of the area to fill.

    color : color
        The color of the area to fill.

    alpha : float, optional (default=0.5)

    '''

    # bins gives the bin centers, we need the bin edges: [1,1, 3,3, 7,7]
    _x = np.repeat(bins, 2)[1:-1]

    # count each errorbar twice
    _y1 = np.repeat(y1, 2)
    _y2 = np.repeat(y2, 2)

    # fill histogram errorbars
    axes.fill_between(
        _x,
        _y1,
        _y2,
        color=color,
        alpha=alpha
    )
    return


def plot_oob_marker(axes, x, y, upper_bound=None, lower_bound=None, color=None):
    '''Plots out of bound markers at the upper or lower boundary.

    Parameters:
    -----------
    ax : matplotlib.pyplot.axis
        Where to pot the data.

    x : array-like
        The horizontal coordinates of the data points.

    y : array-like
        The vertical coordinates of the data points.

    upper_bound : float, optional (default=None)
        Upper limit of the y axis.

    lower_bound : float, optional (default=None)
        Lower limit of the y axis.

    color : color, optional (default=None)
        A color for the triangles.
        See following link for more information:
        https://matplotlib.org/3.3.2/tutorials/colors/colors.html'''
    # Plot upper triangles
    if upper_bound is not None:
        upper_mask = np.array(y) > upper_bound
        axes.plot(x[upper_mask], [upper_bound]*sum(upper_mask), linestyle='', marker=6, color=color)
    # Plot lower triangles
    if lower_bound is not None:
        lower_mask = y < lower_bound
        axes.plot(x[lower_mask], [lower_bound]*sum(lower_mask), linestyle='', marker=7, color=color)
    return

def plot_ratio_result_true(target_bins, R_july_reco, err_july, R_dez_reco, err_dez, R_july_true, R_dez_true, xlabel, path_out, lower_bound_ratio=None, upper_bound_ratio=None):
    # main plot
    #bincenters = target_bins[:-1] + np.diff(target_bins)/2
    
    # calc bin centers on log scale
    bincenters = np.exp((np.log(target_bins[:-1]) + np.log(target_bins[1:]))/2)

    xerr = np.diff(target_bins)/2
    

    fig, (axes1, axes2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [4,1]})
    # main plot
    axes1.plot(
        [target_bins[0], target_bins[-1]],
        [1,1],
        'k-'
    )
    #axes1.errorbar(
    #    bincenters,
    #    R_july_reco,
    #    xerr=xerr,
    #    yerr=err_july,
    #    marker='D',
    #    ls='',
    #    label='July reco.',
    #    color='#84B819'
    #)

    # july reco + true
    axes1.plot(
        bincenters,
        R_july_reco,
        marker='p',
        ls='',
        label='July reco.',
        color='#84B819'
    )
    fill_between_axes(axes1, target_bins, R_july_reco - err_july[0], R_july_reco + err_july[1], color='#84B819')

    axes1.hist(
        bincenters,
        bins=target_bins,
        weights=R_july_true,
        histtype='step',
        label='July true',
        color='#84B819',
        linestyle=':'
    )

    # dez reco + true
    axes1.plot(
        bincenters,
        R_dez_reco,
        marker='p',
        ls='',
        label='Dezember reco.',
        color='#E36913'
    )
    fill_between_axes(axes1, target_bins, R_dez_reco - err_dez[0], R_dez_reco + err_dez[1], color='#E36913')
    
    axes1.hist(
        bincenters,
        bins=target_bins,
        label='Dezember true',
        weights=R_dez_true,
        histtype='step',
        color='#E36913',
        linestyle=':'
    )

    axes1.set_ylabel(r'$R = $season / year')
    #axes1.set_yscale('log')
    axes1.set_xscale('log')
    axes1.legend()

    # ratio plot
    axes2.plot(
        [target_bins[0], target_bins[-1]],
        [1,1],
        'k-'
    )
    #axes2.errorbar(
    #    bincenters,
    #    R_july_reco/R_july_true,
    #    xerr=xerr,
    #    yerr=[err_july[0]/R_july_true, err_july[1]/R_july_true],
    #    marker='D',
    #    ls='',
    #    color='#84B819'
    #)

    # july reco / true
    axes2.plot(
        bincenters,
        R_july_reco/R_july_true,
        marker='p',
        ls='',
        color='#84B819'
    )
    fill_between_axes(axes2, target_bins, R_july_reco/R_july_true - err_july[0]/R_july_true, R_july_reco/R_july_true + err_july[1]/R_july_true, color='#84B819')
    
    # dez reco / true
    axes2.plot(
        bincenters,
        R_dez_reco/R_dez_true,
        marker='p',
        ls='',
        color='#E36913'
    )
    fill_between_axes(axes2, target_bins, R_dez_reco/R_dez_true - err_dez[0]/R_dez_true, R_dez_reco/R_dez_true + err_dez[1]/R_dez_true, color='#E36913')


    axes2.set_xlabel(xlabel)
    axes2.set_ylabel(r'$R_{\mathrm{reco.}} / R_{\mathrm{true}}$')
    axes2.set_xlim(target_bins[0], target_bins[-1])

    # set ylim/zoom if bounds are given
    if lower_bound_ratio and upper_bound_ratio:
        axes2.set_ylim(lower_bound_ratio, upper_bound_ratio)
        plot_oob_marker(axes2, bincenters, R_july_reco/R_july_true, upper_bound=upper_bound_ratio, lower_bound=lower_bound_ratio, color='#84B819')
        plot_oob_marker(axes2, bincenters, R_dez_reco/R_dez_true, upper_bound=upper_bound_ratio, lower_bound=lower_bound_ratio, color='#E36913')

    plt.tight_layout()
    plt.savefig(path_out)
    return
    
def plot_ratio_season_year(target_bins, true_target, est_annual, err_annual, est_12, err_12, est_14, err_14, xlabel, path_out, lower_bound_ratio=None, upper_bound_ratio=None):
    bincenters = target_bins[:-1] + np.diff(target_bins)/2
    xerr = np.diff(target_bins)/2

    fig, (axes1, axes2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [4,1]})
    # main plot
    axes1.errorbar(
        bincenters,
        est_annual,
        xerr=xerr,
        yerr=err_annual,
        marker='.',
        ls='',
        label='Year',
        color='cornflowerblue'
    )
    axes1.step(
        target_bins,
        np.append(true_target,0),
        where='post',
        label='True',
        color='black',
        linestyle=':'
    )
    axes1.errorbar(
        bincenters,
        est_12,
        xerr=xerr,
        yerr=err_12,
        marker='.',
        ls='',
        label='July',
        color='#84B819'
    )
    axes1.errorbar(
        bincenters,
        est_14,
        xerr=xerr,
        yerr=err_14,
        marker='.',
        ls='',
        label='Dezember',
        color='#E36913'
    )
    axes1.set_ylabel(r'Event rate / s')
    axes1.set_yscale('log')
    axes1.set_xscale('log')
    axes1.legend()

    # ratio plot
    axes2.plot(
        [target_bins[0], target_bins[-1]],
        [1,1],
        'k-'
    )
    axes2.errorbar(
        bincenters,
        est_12/est_annual,
        xerr=xerr,
        yerr=[ratio_error(est_12, est_annual, err_12[0], err_annual[0]), ratio_error(est_12, est_annual, err_12[1], err_annual[1])],
        marker='.',
        ls='',
        color='#84B819'
    )
    axes2.errorbar(
        bincenters,
        est_14/est_annual,
        xerr=xerr,
        yerr=[ratio_error(est_14, est_annual, err_14[0], err_annual[0]), ratio_error(est_14, est_annual, err_14[1], err_annual[1])],
        marker='.',
        ls='',
        color='#E36913'
    )
    # set ylim/zoom if bounds are given
    if lower_bound_ratio and upper_bound_ratio:
        axes2.set_ylim(lower_bound_ratio, upper_bound_ratio)
        plot_oob_marker(axes2, bincenters, est_12/est_annual, upper_bound=upper_bound_ratio, lower_bound=lower_bound_ratio, color='#84B819')
        plot_oob_marker(axes2, bincenters, est_14/est_annual, upper_bound=upper_bound_ratio, lower_bound=lower_bound_ratio, color='#E36913')

    axes2.set_xlabel(xlabel)
    axes2.set_ylabel('season /\nyear')
    axes2.set_xlim(target_bins[0], target_bins[-1])
    plt.tight_layout()
    plt.savefig(path_out)
