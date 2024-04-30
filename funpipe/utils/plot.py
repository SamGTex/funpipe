import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from .calc import ratio_error
from .calc import weighted_pearson_corr, weighted_cov
from .calc import calc_weighted_quantiles_per_bin, calc_Wmean_Wstd

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

# custom colors
TUORANGE = (227/255, 105/255, 19/255)
TUGREEN = (132/255, 184/255, 25/255)
DARK = '#3a3d41'
DARKBLUE = '#170a45'

# ------ helper functions ------
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


def plot_errorend_bins(ax, bins, mean, lower_err, upper_err, color='black', linewidth=1, logx=False,width=0.2):
    '''
    Plot horizontal line at end of errorbars.

    Parameters:
    -----------
    ax : matplotlib.pyplot.axis
        Where to pot the data.
    bins: numpy.array
        The bin edges of the histogram.
    mean : numpy.array
        The mean values of the histogram.
    lower_err : numpy.array
        The lower error of the histogram.
    upper_err : numpy.array
        The upper error of the histogram.
    color : color, optional (default='black')
        The color of the errorbars.
    linewidth : float, optional (default=1)
        The width of the errorbars.
    logx : bool, optional (default=False)
        If True, the x-axis is logarithmic.

    Returns:
    --------
    ax : matplotlib.pyplot.axis
        The axis with the errorbars.
    '''

    if logx == False:
        for i, bin_value in enumerate(bins[:-1]):
            bin_width = bins[i+1] - bins[i]
            bin_center = bin_value + 0.5*bin_width
            err_start = bin_center + 0.5*bin_width
            err_end = bin_center - 0.5*bin_width

            ax.hlines(mean[i]-lower_err[i], err_start, err_end, color=color, linewidth=linewidth)
            ax.hlines(mean[i]+upper_err[i], err_start, err_end, color=color, linewidth=linewidth)

    else:
        for i, bin_value in enumerate(bins[:-1]):
            log_bin_start = np.log10(bins[i])
            log_bin_end = np.log10(bins[i+1])

            # error half of bin
            err_start = 10**((log_bin_start + log_bin_end)/2 + width/2*(log_bin_end - log_bin_start))
            err_end = 10**((log_bin_start + log_bin_end)/2 - width/2*(log_bin_end - log_bin_start))

            ax.hlines(mean[i]-lower_err[i], err_start, err_end, color=color, linewidth=linewidth)
            ax.hlines(mean[i]+upper_err[i], err_start, err_end, color=color, linewidth=linewidth)



def plot_errorend(ax, bin_centers, mean, lower_err, upper_err, color='black', linewidth=1):
    '''
    Plot horizontal line at end of errorbars.

    Parameters:
    -----------
    ax : matplotlib.pyplot.axis
        Where to pot the data.
    bin_centers : numpy.array
        The bin centers of the histogram.
    mean : numpy.array
        The mean values of the histogram.
    lower_err : numpy.array
        The lower error of the histogram.
    upper_err : numpy.array
        The upper error of the histogram.
    color : color, optional (default='black')
        The color of the errorbars.
    linewidth : float, optional (default=1)
        The width of the errorbars.

    Returns:
    --------
    ax : matplotlib.pyplot.axis
        The axis with the errorbars.
    '''

    for i, bin_value in enumerate(bin_centers):
        err_start = bin_value - 0.5*(bin_centers[1] - bin_centers[0])
        err_end = bin_value + 0.5*(bin_centers[1] - bin_centers[0])

        ax.hlines(mean[i]-lower_err[i], err_start, err_end, color=color, linewidth=linewidth)
        ax.hlines(mean[i]+upper_err[i], err_start, err_end, color=color, linewidth=linewidth)
    
    return ax

def plot_ratio_result_true(target_bins, R_july_reco, err_july, R_dez_reco, err_dez, R_july_true, R_dez_true, xlabel, path_out, lower_bound_ratio=None, upper_bound_ratio=None, dark_mode=False):
    # check if dark mode is set
    if dark_mode:
        plt.style.use('dark_background')
        facecolor = DARK
    else:
        plt.style.use('default')
        facecolor = 'white'
    # main plot
    #bincenters = target_bins[:-1] + np.diff(target_bins)/2
    
    # calc bin centers on log scale
    bincenters = np.exp((np.log(target_bins[:-1]) + np.log(target_bins[1:]))/2)
    xerr = np.diff(target_bins)/2
    
    fig, (axes1, axes2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [4,1]}, facecolor=facecolor)
    
    # Set face color for each subplot
    axes1.set_facecolor(facecolor)
    axes2.set_facecolor(facecolor)

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
        label='January reco.',
        color='#E36913'
    )
    fill_between_axes(axes1, target_bins, R_dez_reco - err_dez[0], R_dez_reco + err_dez[1], color='#E36913')
    
    axes1.hist(
        bincenters,
        bins=target_bins,
        label='January true',
        weights=R_dez_true,
        histtype='step',
        color='#E36913',
        linestyle=':'
    )

    axes1.set_ylabel(r'$R = $season / year')
    #axes1.set_yscale('log')
    axes1.set_xscale('log')
    axes1.legend(facecolor=facecolor)

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
    plt.savefig(path_out, facecolor=facecolor)
    return
    
def plot_ratio_season_year(target_bins, true_target, est_annual, err_annual, est_12, err_12, est_14, err_14, xlabel, path_out, lower_bound_ratio=None, upper_bound_ratio=None, dark_mode=False, y_label=None):
    if dark_mode:
        plt.style.use('dark_background')
        facecolor = DARK
        color_yearunf = 'white'#'#d81b60'
        color_yeartrue = 'white'
    else:
        plt.style.use('default')
        facecolor = 'white'
        color_yearunf = DARKBLUE
        color_yeartrue = 'black'

    if y_label is None:
        y_label = r'Event Rate / Hz'

    bincenters = target_bins[:-1] + np.diff(target_bins)/2
    
    #bincenters = np.exp((np.log(target_bins[:-1]) + np.log(target_bins[1:]))/2)

    xerr = np.diff(target_bins)/2
    fig, (axes1, axes2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [4,1]}, facecolor=facecolor)
    
    # Set face color for each subplot
    axes1.set_facecolor(facecolor)
    axes2.set_facecolor(facecolor)
    
    # main plot
    axes1.errorbar(
        bincenters,
        est_annual,
        xerr=xerr,
        yerr=err_annual,
        marker='.',
        ls='',
        label='Year reco.',
        color=color_yearunf
    )
    axes1.step(
        target_bins,
        np.append(true_target,0),
        where='post',
        label='year true',
        color=color_yeartrue,
        linestyle=':'
    )
    axes1.errorbar(
        bincenters,
        est_12,
        xerr=xerr,
        yerr=err_12,
        marker='.',
        ls='',
        label='July reco.',
        color='#84B819'
    )
    axes1.errorbar(
        bincenters,
        est_14,
        xerr=xerr,
        yerr=err_14,
        marker='.',
        ls='',
        label='January reco.',
        color='#E36913'
    )
    axes1.set_ylabel(y_label)
    axes1.set_yscale('log')
    axes1.set_xscale('log')
    axes1.legend(facecolor=facecolor)

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
    plt.savefig(path_out, facecolor=facecolor)

# -----------------------------
# PLOTTER: 2d-distribution plot
def plot_distr_simple(x, y, weights, xlabel, ylabel, path, Nbins, xlim=(None,None)):
    fig = plt.figure(figsize=(8,6), dpi=300)
    _, _, _, pcm = plt.hist2d(x, y, weights=weights, bins=Nbins, cmap='jet')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
  
    if xlim[0]!=None and xlim[1]!=None: #set xlim if not None      
        print(xlim)
        plt.xlim(*xlim)

    plt.colorbar(pcm, label = 'Event Rate / Hz')
    plt.savefig(path)
    plt.close(fig)

def plot_distr(x, y, weights, xlabel, ylabel, path_out, Nbins, figsize, cmap, fontsize=18, xlim=(None, None), ylim=(None,None), show_corr=False):
    # Use a light theme
    plt.style.use('default')
    plt.rcParams.update({'font.size': fontsize})

    # Create figure and grid
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(3, 3, width_ratios=[1, 0.1, 0.05], height_ratios=[0.1, 1, 0.1])

    # Central 2D Histogram
    ax_center = fig.add_subplot(gs[1, 0])
    pcm = ax_center.hist2d(x, y, weights=weights, bins=Nbins, cmap=cmap, norm=None)[3]#LogNorm()

    # Top histogram (x-axis)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_center)
    ax_top.hist(x, bins=Nbins, weights=weights, histtype='stepfilled', linewidth=3, edgecolor='black', color=TUORANGE)
    ax_top.tick_params(axis='y', labelleft=False)
    ax_top.tick_params(axis='x', labelbottom=False)

    # Right histogram (y-axis)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_center)
    ax_right.hist(y, bins=Nbins, weights=weights, orientation='horizontal', histtype='stepfilled', linewidth=3, edgecolor='black', color=TUORANGE)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelbottom=False)

    # calc pearson correlation if show_corr=True
    if show_corr:
        corr = weighted_pearson_corr(x, y, weights)
        ax_center.text(0.05, 0.9, f'Pearson correlation: {corr:.2f}', transform=ax_center.transAxes, color='white')

    # Set labels
    ax_center.set_xlabel(xlabel)
    ax_center.set_ylabel(ylabel)

    # Set xlim if not None
    if xlim[0] is not None and xlim[1] is not None:
        ax_center.set_xlim(*xlim)

    # Set ylim if not None
    if ylim[0] is not None and ylim[1] is not None:
        ax_center.set_ylim(*ylim)

    # Add colorbar to the right
    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(pcm, cax=cbar_ax, label=r'Event Rate / Hz')

    # Adjust layout
    plt.tight_layout(h_pad=-1.5)#h_pad=-1.5, w_pad=-0.2)

    # Save the figure
    plt.savefig(path_out)
    plt.close(fig)

def plot_distr_dark(x, y, weights, xlabel, ylabel, path_out, Nbins, figsize, cmap, fontsize=18, xlim=(None, None), ylim=(None, None), show_corr=False):
    # Use a dark theme
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': fontsize})

    # Create figure and grid
    fig = plt.figure(figsize=figsize, dpi=300, facecolor=DARKBLUE)
    gs = GridSpec(3, 3, width_ratios=[1, 0.1, 0.05], height_ratios=[0.1, 1, 0.1])

    # Central 2D Histogram
    ax_center = fig.add_subplot(gs[1, 0])
    pcm = ax_center.hist2d(x, y, weights=weights, bins=Nbins, cmap=cmap, norm=None)[3]#LogNorm()
    
    # Top histogram (x-axis)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_center)
    

    ax_top.hist(x, bins=Nbins, weights=weights, histtype='stepfilled', linewidth=3, edgecolor='black', color=TUORANGE)
    ax_top.tick_params(axis='y', labelleft=False)
    ax_top.tick_params(axis='x', labelbottom=False)

    # Right histogram (y-axis)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_center)
    ax_right.hist(y, bins=Nbins, weights=weights, orientation='horizontal', histtype='stepfilled', linewidth=3, edgecolor='black', color=TUORANGE)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelbottom=False)

    # Calculate Pearson correlation if show_corr=True
    if show_corr:
        corr = weighted_pearson_corr(x, y, weights)
        ax_center.text(0.05, 0.9, f'Pearson correlation: {corr:.2f}', transform=ax_center.transAxes, color='white')

    # Set labels
    ax_center.set_xlabel(xlabel, color='white')
    ax_center.set_ylabel(ylabel, color='white')

    # Set xlim if not None
    if xlim[0] is not None and xlim[1] is not None:
        ax_center.set_xlim(*xlim)

    # Set ylim if not None
    if ylim[0] is not None and ylim[1] is not None:
        ax_center.set_ylim(*ylim)

    # Add colorbar to the right
    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(pcm, cax=cbar_ax, label=r'Event Rate / Hz')

    # Set color of ticks and labels
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Adjust layout
    plt.tight_layout(h_pad=-1.5)#h_pad=-1.5, w_pad=-0.2)

    # Save the figure
    plt.savefig(path_out, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # set back to default
    plt.style.use('default')

def cmap_custom(color_list):
    cmap = LinearSegmentedColormap.from_list('Custom_Blend', color_list)
    return cmap

def cmap_tuorange():
    return cmap_custom([DARKBLUE, TUORANGE, (1,1,1)])

def cmap_tuorange_reverse():
    return cmap_custom([(1,1,1), TUORANGE, DARKBLUE])

# -----------------------------
# MC vs. Data comparison
def datamc_plot(bins, livetime, df_data, df_mc, w_name_mc, w_name_plot, w_colors, var, xlabel, filename, xlog=False, dark=False, lower=0.5, upper=1.5, fontsize=18):
    '''
    Plot data and MC comparison in event rate and ratio.

    Parameters:
    -----------
    bins : array-like
        The bin edges of the histogram.

    livetime : float
        The livetime of the data in seconds.

    df_data : pandas.DataFrame
        The measured data.

    df_mc : pandas.DataFrame
        The simulated data.

    w_name_mc : list
        The column names in df_mc of the MC weights.

    w_name_plot : list
        The names of the weights for the legend.

    w_colors : list
        The colors for each weight.

    var : str
        The column name of the variable to plot.

    xlabel : str
        The label of the x-axis.

    filename : str
        The path to save the plot.

    xlog : bool, optional (default=False)
        If True, the x-axis is logarithmic.

    dark : bool, optional (default=False)
        If True, a dark theme is used.

    lower : float, optional (default=0.5)
        The lower bound of the ratio plot.

    upper : float, optional (default=1.5)
        The upper bound of the ratio plot.

    fontsize : int, optional (default=18)
        The fontsize of all labels and ticks.

    Returns:
    --------
    None
    '''

    # theme
    if dark:
        plt.style.use('dark_background')
        color_data = 'white'
    else:
        plt.style.use('default')
        color_data = 'black'
    plt.rcParams.update({'font.size': fontsize})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # reduce place between subplots
    plt.subplots_adjust(hspace=0.0)
    
    # log scale
    if xlog:
        bincenters = np.power(10, (np.log10(bins[:-1]) + np.log10(bins[1:]))/2)
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    else:
        bincenters = (bins[:-1] + bins[1:])/2
        
    # plot data
    hist_data, bins_data = np.histogram(df_data[var], bins=bins)
    error_data = np.sqrt(hist_data)
 
    rate_data = hist_data / livetime
    error_data = error_data / livetime
    
    ax1.errorbar(bincenters, rate_data, yerr=error_data, fmt='o', color=color_data, label='Data')

    # plot MC
    for _weight, _color, _label in zip(w_name_mc, w_colors, w_name_plot):
        # plot histogram in main
        hist_mc, bins_mc = np.histogram(df_mc[var], bins=bins, weights=df_mc[_weight])
        ax1.step(bins_mc, np.append(hist_mc, hist_mc[-1]), where='post', color=_color, label=_label)
        
        # plot error in main
        mc_err= (np.sqrt(np.histogram(df_mc[var], bins=bins, weights=df_mc[_weight]**2)[0]))
        ax1.errorbar(bincenters, hist_mc, yerr=mc_err, fmt='none', color=_color)

        # plot ratio
        ratio_mean = rate_data / hist_mc
        ratio_err = ratio_error(rate_data, hist_mc, error_data, mc_err)
        ax2.errorbar(bincenters, ratio_mean, yerr=ratio_err, fmt='o', color=_color)
        plot_oob_marker(ax2, bincenters, ratio_mean, upper, lower, _color)

    # horizontal line at 1
    ax2.axhline(1, color=color_data, linestyle='-', linewidth=2)

    ax1.set_yscale('log')
    ax1.set_ylabel('Event Rate / Hz')
    ax1.legend(fontsize=fontsize-3)

    ax2.set_ylabel('Data / MC')
    ax2.set_xlabel(xlabel)

    # set bounds
    ax2.set_ylim(lower, upper)
    ax2.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])

    # set xlim
    ax1.set_xlim(bins[0], bins[-1])

    plt.savefig(filename, dpi=300, transparent=True)

    return

# -----------------------------
# Variable x vs. mean of variable y in histogram with ratio plot and mask option
def plot_x_vs_ymean(bins_x, x, y, mask_list, mask_names, mask_colors, weights, xlabel, ylabel, path_out, plot_error=False, xlog=True, ylog=True, theme='light', use_quantiles=True, fontsize=16, y_limit_low=0.85, y_limit_upper=1.15):
    '''
    Plot 1D-Histogram for variable x vs. mean of variable y.

    Parameters
    ----------
    bins_x: np.array
        Binning in x
    x: np.array
        Data for x
    y: np.array
        Data for y
    mask_list: list(np.array)
        List of masks for different samples
    mask_names: list(str)
        List of names for different samples, used as label
    mask_colors: list(str)
        List of colors for different samples, used for plotting
    weights: np.array
        Weights for each event
    xlabel: str
        Label for x-axis
    ylabel: str
        Label for y-axis
    path_out: str
        Path to save the plot
    plot_error: bool
        Plot error bars if True.
    xlog: bool
        Log scale for x-axis if True.
    ylog: bool
        Log scale for y-axis if True.
    theme: str
        Theme for plotting. 'light' or 'dark'
    use_quantiles: bool
        Use median and 1 sigma quantiles if True, else mean and std.
    fontsize: int
        Fontsize for plot
    y_limit_low: float
        Lower bound for y-axis
    y_limit_upper: float
        Upper bound for y-axis

    Returns
    -------
    None
    '''
    
    
    if theme == 'light':
        plt.style.use('default')
        plt.rcParams.update({'font.size': fontsize})
        color_alldata = 'k'

    else:
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': fontsize})
        color_alldata = 'w'

    quantiles = [0.16, 0.5, 0.84]

    #binning
    #bins_x = np.geomspace(x.min(), x.max(), nbins_x+1)
    bin_centers = np.array([(bins_x[i] + (bins_x[i+1]-bins_x[i])/2) for i in range(len(bins_x)-1)])
    bin_width = np.array([bins_x[i+1]-bins_x[i] for i in range(len(bins_x)-1)])
    bin_mid_onlog = np.array([10**((np.log10(bins_x[i]) + (np.log10(bins_x[i+1])-np.log10(bins_x[i]))/2)) for i in range(len(bins_x)-1)])

    #plot
    fig = plt.figure(figsize=(8,6),dpi=300) #dpi 300 for full hd
    fig.tight_layout()
    gs = matplotlib.gridspec.GridSpec(4, 1)
    axes1 = fig.add_subplot(gs[:-1])
    axes2 = fig.add_subplot(gs[-1], sharex=axes1)
    fig.subplots_adjust(hspace = .1) #0.001

    # calc weighted y mean+std for each log energy bin
    print(f'\n{x.shape[0]} events in total:')
    if use_quantiles:
        _quantiles_alldata = calc_weighted_quantiles_per_bin(x, y, weights, bins_x)
        atm_mean, atm_std = _quantiles_alldata[:,1], np.column_stack((_quantiles_alldata[:,0], _quantiles_alldata[:,2])).T
    else:
        atm_mean, atm_std = calc_Wmean_Wstd(x, y, weights, bins_x)

    atm_mean_list = []
    atm_std_list = []
    ratio_mean_list = []
    ratio_std_list = []
    for mask, mask_name in zip(mask_list, mask_names):
        print(f'\n{mask.sum()} events for {mask_name}:')
        print(f'Event rate: {weights[mask].sum()} Hz')

        if use_quantiles:
            _quantiles = calc_weighted_quantiles_per_bin(x[mask], y[mask], weights[mask], bins_x)
            _atm_mean, _atm_std = _quantiles[:,1], np.column_stack((_quantiles[:,0], _quantiles[:,2])).T
        else:
            _atm_mean, _atm_std = calc_Wmean_Wstd(x[mask], y[mask], weights[mask], bins_x)

        # mean and std
        atm_mean_list.append(_atm_mean)
        atm_std_list.append(_atm_std)

        # ratio
        ratio_mean_list.append(_atm_mean/atm_mean)
        ratio_std_list.append(np.sqrt((_atm_std/atm_mean)**2 + (atm_std*_atm_mean/atm_mean**2)**2))

    # set error if true
    if plot_error and use_quantiles:
        yearly_std = atm_std
    else:
        yearly_std = None

    # plot all data mean
    axes1.errorbar(bin_centers, atm_mean, fmt=' ', color=color_alldata, xerr=bin_width/2, yerr=yearly_std, elinewidth=2, label='all data') #true distr
    

    # plot mean for each given mask
    for _atm_mean, _atm_std, _atm_name, _color in zip(atm_mean_list, atm_std_list, mask_names, mask_colors):

        # set error if true
        if plot_error:
            _atm_i_std = _atm_std
        else:
            _atm_i_std = None

        axes1.errorbar(bin_centers, _atm_mean, fmt=' ', xerr=bin_width/2, yerr=_atm_i_std, elinewidth=1, label=_atm_name, color=_color)

    # set log scale if true
    if xlog:
        axes1.set_xscale('log')
        axes2.set_xscale('log')
    if ylog:
        axes1.set_yscale('log')
    #axes1.set_xticks([],[])
    axes1.set_ylabel(ylabel)


    # ratio
    for _ratio_mean, _ratio_std, _atm_name, _color in zip(ratio_mean_list, ratio_std_list, mask_names, mask_colors):

        # set error if true
        if plot_error:
            _ratio_i_std = _ratio_std
        else:
            _ratio_i_std = None

        axes2.errorbar(bin_centers, _ratio_mean, fmt=' ', xerr=bin_width/2, yerr=_ratio_i_std, elinewidth=1, label=_atm_name, color=_color)
        plot_oob_marker(axes2, bin_centers, _ratio_mean, lower_bound=y_limit_low, upper_bound=y_limit_upper, color=_color)

    axes2.axhline(1, color=color_alldata, linestyle='--', linewidth=1)
    axes2.set_xlabel(xlabel)
    axes2.set_ylim(y_limit_low, y_limit_upper)
    axes2.set_yticks([y_limit_low,1.0,y_limit_upper])
    axes2.set_ylabel('Ratio')

    # disable xticks in top plot
    axes1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # legend right side next to plot
    axes1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # set xlim to min and max of energy
    axes1.set_xlim(bins_x[0], bins_x[-1])

    plt.savefig(path_out, transparent=True)
    plt.close(fig)