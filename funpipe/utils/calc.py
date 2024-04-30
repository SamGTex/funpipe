import numpy as np
from scipy.stats import chi2

def ratio_error(mean_numerator, mean_denominator, std_numerator, std_denominator):
    '''
    Caculate the error of a ratio by Gaussian Uncertainty Propagation.

    Parameters
    -----------
    mean_numerator : array, size(N,), floats
        The mean value of the numerator array.

    mean_denominator : array, size(N,), floats
        The mean value of the denominator array.

    std_numerator : array, size(N,), floats
        The uncertainty value of the numerator array.

    std_denominator : array, size(N,), floats
        The uncertainty value of the denominator array.

    Returns
    ----------
    ratio_err : array, size(N,), floats
        The uncertainty of the ratio.

    '''
    #cov = np.cov(mean_numerator[4],mean_denominator[4])
    #print(mean_numerator)
    #print('cov=', cov)
    ratio_err = np.sqrt((std_numerator/mean_denominator)**2 + (- (mean_numerator * std_denominator)/(mean_denominator)**2)**2)
    #ratio_err = np.sqrt((std_numerator/mean_denominator)**2 + (- (mean_numerator * std_denominator)/(mean_denominator)**2)**2 + 2*(std_numerator/mean_denominator)*(- (mean_numerator * std_denominator)/(mean_denominator))*cov[0,1])

    return ratio_err

def chi2_test(ratio_unf, ratio_true, ratio_err):
    '''
    Caculate the chi2 value and pvalue of unfolded ratio to true ratio.

    Parameters
    -----------
    ratio_unf : array, size(N,), floats
        The unfolded ratio.
    
    ratio_true : array, size(N,), floats
        The true ratio.

    ratio_err : array, size(N,), floats
        The uncertainty of the unfolded ratio.
    
    Returns
    ----------
    chi2 : float
        The chi2 value.
    
    pvalue : float
        The pvalue.
    '''

    expected_data = np.ones(len(ratio_unf))
    observed_data = ratio_unf/ratio_true
    
    # chi2 value
    chi2value = 0.
    for i in range(len(expected_data)):
        print(chi2value)
        chi2value += ((observed_data[i])-(expected_data[i]))**2./((ratio_err[i]/ratio_true[i])**2)
    print('Chi2 value:',chi2value)

    # pvalue
    pvalue =  1-chi2.cdf(chi2value, len(expected_data)-1)
    print('Pvalue = ', pvalue)
    
    return chi2value, pvalue

def weighted_cov(x, y, w):
    """Weighted Covariance"""
    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)
    return np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w)


def weighted_pearson_corr(x, y, w=None):
    """Weighted Pearson Correlation"""

    if w is None:
        return np.corrcoef(x, y)[0][1]
        # w = np.ones_like(x)

    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def calc_weighted_quantiles_per_bin(x, y, weights, bins):
    quantiles = [0.16, 0.5, 0.84]

    quantiles_per_bin = []

    for i in range(len(bins)-1):
        mask = (x > bins[i]) & (x < bins[i+1])
        x_bin = x[mask]
        y_bin = y[mask]
        weights_bin = weights[mask]

        quantiles_bin = weighted_quantile(y_bin, quantiles, sample_weight=weights_bin)
        quantiles_per_bin.append(quantiles_bin)

    return np.array(quantiles_per_bin)

def calc_Wmean_Wstd(E, multiplicity, weights, logbins):
    # calc mean and std for each log energy bin
    mean = []
    std = []
    print(logbins)
    for i in range(len(logbins)-1):
        mask = (E>=logbins[i]) & (E<logbins[i+1])
        print(f'{mask.sum()} events in bin {i}: {logbins[i]} - {logbins[i+1]}')
        mean.append(np.average(multiplicity[mask], weights=weights[mask]))

        factor = (len(multiplicity[mask] - 1) / len(multiplicity[mask]))
        std.append(np.sqrt(1/factor * np.average((multiplicity[mask]-mean[i])**2, weights=weights[mask])))
    
    bins_mean = np.array(mean)
    bins_std = np.array(std)

    return bins_mean, bins_std