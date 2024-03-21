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