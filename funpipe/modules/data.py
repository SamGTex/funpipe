import pandas as pd
import numpy as np
import simweights
import os
import h5py
import warnings
from tables import NaturalNameWarning
from tqdm import tqdm

# ---------- read in and process data: MC or exp. ----------
class DataManager:
    def __init__(self):
        self.df_raw = pd.DataFrame()
        self.event_type = False
        self.target = None

    def read_mc_weights(self, filelist, path_feature_names, total_files, models, model_names, exp=False):
        '''
        Read in data from hdf5 files and save weights and features (saved in path_feature_names) to dataframe.
        Supports multiple weighting models.

        Parameters
        ----------
        filelist : list
            List of paths to hdf5 files.
        path_feature_names : str
            Path to csv file with feature names.
        total_files : int
            Number of files for calculating weights.
        models : list of simweights.Model
            Models to calculate weights.
        model_names : list of str
            Output name of model in dataframe.
            
        Returns
        -------
        df_raw : pandas.DataFrame
            Dataframe with all features and weights.
        '''
        # get variable names
        df_colnames = pd.read_csv(path_feature_names, comment='#', names=['column', 'subcolumn'], skipinitialspace=True)
        col_names = df_colnames['column'].to_list()
        subcol_names = df_colnames['subcolumn'].to_list()

        # ignore NaturalNameWarning
        warnings.filterwarnings('ignore', category=NaturalNameWarning)

        # read in files
        for file in tqdm(filelist):
            #print(f'\rReading file {file}...', end='')
            _df = pd.DataFrame()

            # read in hdf5 file and store as Weighter object
            _hdf = pd.HDFStore(file, "r")
            weighter = simweights.CorsikaWeighter(_hdf,total_files)
            
            # calculate and save weights in df
            for model, model_name in zip(models, model_names):
                _df[model_name] = weighter.get_weights(model)

            # save features in df
            for colname, subcolname in zip(col_names, subcol_names):
                #print(f'Write {colname}.{subcolname} in DataFrame.')
                _df[colname+"."+subcolname] = weighter.get_column(colname, subcolname)

            # append to dataframe
            self.df_raw = pd.concat([self.df_raw, _df], ignore_index=True)

            _hdf.close()

        print('Done.\n')
        
        return self.df_raw
    
    def read_mc(self, filelist, path_feature_names, total_files, model, exp=False):
        '''
        Read in data from hdf5 files and save weights and features (saved in path_feature_names) to dataframe.
        Supports only one weighting model.

        Parameters
        ----------
        filelist : list
            List of paths to hdf5 files.
        path_feature_names : str
            Path to csv file with feature names.
        total_files : int
            Number of files for calculating weights.
        model : simweights.Model
            Model to calculate weights.
            
        Returns
        -------
        df_raw : pandas.DataFrame
            Dataframe with all features and weights.
        '''
        # get variable names
        df_colnames = pd.read_csv(path_feature_names, comment='#', names=['column', 'subcolumn'], skipinitialspace=True)
        col_names = df_colnames['column'].to_list()
        subcol_names = df_colnames['subcolumn'].to_list()

        # ignore NaturalNameWarning
        warnings.filterwarnings('ignore', category=NaturalNameWarning)

        # get weights and write in dataframe
        for file in tqdm(filelist):
            # refresh line
            #print(f'\rReading file {file}...', end='')
            _hdf = pd.HDFStore(file, "r")

            weighter = simweights.CorsikaWeighter(_hdf,total_files)

            _df = pd.DataFrame()
            _df["weights"] = weighter.get_weights(model)


            # save all variables in df
            for colname, subcolname in zip(col_names, subcol_names):
                #print(f'Write {colname}.{subcolname} in DataFrame.')
                _df[colname+"."+subcolname] = weighter.get_column(colname, subcolname)

            # append to dataframe
            self.df_raw = pd.concat([self.df_raw, _df], ignore_index=True)

            _hdf.close()

        print('Done.\n')
        
        return self.df_raw
    
    def read_in_exp(self, filelist, path_features):
        '''
        Read in data from hdf5 files and save features (saved in path_features) to dataframe.

        Parameters
        ----------
        filelist : list
            List of paths to hdf5 files.
        path_features : str
            Path to csv file with feature names. (Column, Subcolumn)

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with all features.
        '''

        # get variable names
        df_colnames = pd.read_csv(path_features, comment='#', names=['column', 'subcolumn'], skipinitialspace=True)
        col_names = df_colnames['column'].to_list()
        subcol_names = df_colnames['subcolumn'].to_list()

        # read in hdf5 files
        df = pd.DataFrame()

        for file in filelist:

            # read in hdf5 file
            df_i = pd.DataFrame()
            try:
                with h5py.File(file, 'r') as f:
                    #print('Reading file: ' + file)

                    # get features
                    for col, subcol in zip(col_names, subcol_names):
                        df_i[col + '.' + subcol] = f[col][subcol]

                    # append to dataframe
                    self.df_raw = pd.concat([self.df_raw, df_i], ignore_index=True)
            except:
                print(f'Error reading file {file} and column {col}.{subcol}.')

        return self.df_raw

    def read_to_df(self, filelist, path_feature_names, total_files, model):
        # OLD, maybe used in one of the first scripts
        '''
        Read in data from hdf5 files and save weights and features (saved in path_feature_names) to dataframe.

        Parameters
        ----------
        filelist : list
            List of paths to hdf5 files.
        path_feature_names : str
            Path to csv file with feature names.
        total_files : int
            Number of files for calculating weights.
        model : simweights.Model
            Model to calculate weights.
            
        Returns
        -------
        df_raw : pandas.DataFrame
            Dataframe with all features and weights.
        '''
        
        # get weights and write in dataframe
        weighter = None

        list_hdf = []
        for file in filelist:
            # refresh line
            print(f'\rReading file {file}...', end='')
            list_hdf.append(pd.HDFStore(file, "r"))
            if weighter is None:
                weighter = simweights.CorsikaWeighter(list_hdf[-1],total_files)
            else:
                weighter += simweights.CorsikaWeighter(list_hdf[-1],total_files)

        df = pd.DataFrame()
        df["weights"] = weighter.get_weights(model)

        # get variable names
        df_colnames = pd.read_csv(path_feature_names, comment='#', names=['column', 'subcolumn'], skipinitialspace=True)
        col_names = df_colnames['column'].to_list()
        subcol_names = df_colnames['subcolumn'].to_list()

        # save all variables in df
        for colname, subcolname in zip(col_names, subcol_names):
            print(f'Write {colname}.{subcolname} in DataFrame.')
            df[colname+"."+subcolname] = weighter.get_column(colname, subcolname)

        # append to dataframe
        self.df_raw = pd.concat([self.df_raw, df], ignore_index=True)

        # close hdf5 files
        for hdf in list_hdf:
            hdf.close()

        print('Done.\n')

        return self.df_raw

    def __eventtype_labelMC__(self, he_muon, etot_bundle, Qmax, Qtot, Esep_value, Qsep_value):
        # definition of leading, bundle and balloon event
        mask_bundle = (he_muon/etot_bundle < Esep_value) #0
        mask_leading = (he_muon/etot_bundle > Esep_value) & (Qmax/Qtot < Qsep_value) #1
        mask_balloon = (Qmax/Qtot > Qsep_value) #2

        y = np.zeros_like(he_muon)
        y[mask_bundle] = 0
        y[mask_leading] = 1
        y[mask_balloon] = 2

        return y
    
    def __eventtype_labelExp__(self, Qmax, Qtot, Qsep_value):
        # definition of balloon and bundle event (contains also leading muons)
        mask_balloon = (Qmax/Qtot > Qsep_value) #0

        y = np.zeros_like(Qmax)
        y[mask_balloon] = 2

        return y
        
    
    def create_eventtype(self, name_Eleading, nameEbundle, nameQmax, nameQtot, cut_E, cut_Q, exp=False):
        '''
        Leading muon: E_i(max) / E_total > cut_E and Q_i(max)/Q_total < cut_Q
        Balloon: E_i(max) / E_total > cut_E and Q_i(max)/Q_total > cut_Q
        Bundle: E_i(max) / E_total < cut_E

        Parameters
        ----------
        cut_E : float
            Cut value for E_ratio.
        cut_Q : float
            Cut value for Q_ratio.

        Returns
        -------
        df_raw['event_type'] : pandas.Series
            Series of event types. Inplace in df_raw.
            Values: 0 (bundle), 1 (leading), 2 (balloon)

        '''
        if exp:
            self.df_raw['event_type'] = self.__eventtype_labelExp__(self.df_raw[nameQmax], self.df_raw[nameQtot], cut_Q)
            self.event_type = True

        else:
            self.df_raw['event_type'] = self.__eventtype_labelMC__(self.df_raw[name_Eleading], self.df_raw[nameEbundle], self.df_raw[nameQmax], self.df_raw[nameQtot], cut_E, cut_Q)
            self.event_type = True

        return self.df_raw['event_type']

    def print_info(self):
        '''
        Print information about number of events and event type.
        '''
        if self.event_type:
            array_types = self.df_raw['event_type'].to_numpy()
            print(f'Number of events: {self.df_raw.shape[0]}')
            print(f'#leading: {(array_types==1).sum()}')
            print(f'#balloon: {(array_types==2).sum()}')
            print(f'#bundle: {(array_types==0).sum()}\n')
        else:
            print('No event type defined yet.')
            print(f'Number of events: {self.df_raw.shape[0]}\n')

        return

    def substitute_str_in_keys(self, old_str, new_str):
        keys = self.df_raw.keys()

        for key in keys:
            if old_str in key:
                new_key = key.replace(old_str, new_str)

                self.df_raw.rename(columns={key: new_key}, inplace=True)
        return

    def apply_cuts(self, conditions, inplace=True):
        '''
        Apply quality cuts to dataframe.

        Parameters
        ----------
        conditions : dict
            Dictionary conditions with keys 'variable' and 'conditions', where the 2nd. conditions is a
            old format: tuple of threshold and operator
            new format: list of dictionaries with keys 'value' and 'operator'.

        '''

        if isinstance(conditions, dict):
            df_ = self.__applycutsdict__(conditions, inplace)
        elif isinstance(conditions, list):
            df_ = self.__applycutslist__(conditions, inplace)
        elif conditions is None:
            return self.df_raw
        else:
            raise ValueError("Invalid condition format.")
            
        if inplace:
            self.df_raw = df_
            self.df_raw.reset_index(drop=True, inplace=True)
            return self.df_raw

        else:
            return df_

    def __applycutsdict__(self, conditions, inplace=True):
        '''
        Apply quality cuts to dataframe.

        Parameters
        ----------
        conditions : dict
            Dictionary of conditions. Key is column name, value is tuple of threshold and operator.
            Example: conditions = {'Qtot_CleanedPulses.value': (1000.0, '>')}
            Possible operators: '<', '>', '=='
        '''
        #mask = self.df_raw.index.copy()  # Initialize with a mask of all True values
        mask = np.full(len(self.df_raw), True, dtype=bool)
        for column, (threshold, operator) in conditions.items():
            if operator == '<':
                mask &= (self.df_raw[column] < threshold)
            elif operator == '>':
                mask &= (self.df_raw[column] > threshold)
            elif operator == '==':
                mask &= (self.df_raw[column] == threshold)
            else:
                raise ValueError(f"Invalid operator: {operator}. Use '<' or '>'.")
            print(f'Apply cut {column} {operator} {threshold}')
            print(f'Number of events: {mask.sum()}\n')

        return self.df_raw[mask]
        
    def __applycutslist__(self, conditions, inplace=True):
        '''
        Apply quality cuts to dataframe.
        
        Parameters
        ----------
        conditions : list
            List of conditions. Each condition is a dictionary with keys 'variable' and 'conditions_var'. Where conditions_var is a list of dictionaries with keys 'value' and 'operator'.

        '''

        mask = np.full(len(self.df_raw), True, dtype=bool)
        for condition in conditions:
            variable = condition['variable']
            conditions_var = condition['conditions']
            for cond in conditions_var:
                threshold = cond['value']
                operator = cond['operator']
                if operator == '<':
                    mask &= (self.df_raw[variable] < threshold)
                elif operator == '>':
                    mask &= (self.df_raw[variable] > threshold)
                elif operator == '==':
                    mask &= (self.df_raw[variable] == threshold)
                else:
                    raise ValueError(f"Invalid operator: {operator}. Use '<', '>', or '=='.")
                print(f'Apply cut {variable} {operator} {threshold}')
                print(f'Number of events: {mask.sum()}\n')
        
        return self.df_raw[mask]

    def drop_eventtype(self, event_id):
        '''
        Drop events of specific event type.
        0: bundle
        1: leading
        2: balloon

        Parameters
        ----------
        event_id : int
            Event type to drop.

        '''
        if self.event_type:
            self.df_raw = self.df_raw.drop(self.df_raw[self.df_raw['event_type']==event_id].index)
            self.df_raw.reset_index(drop=True, inplace=True)
        else:
            print('No event type defined yet.\nRun create_event_type(ERATIO_CUT, QRATIO_CUT) first.')
        
        return self.df_raw

    def get_df(self):
        return self.df_raw
# ---------- helper functions ----------

# create list of paths to hdf5 files, each HDF5 contains 1000 files
def create_path_list(dir_path, file_prefix, total_files, files_per_hdf, filetype=''):
    '''
    Create list of paths to hdf5 files, each HDF5 contains files_per_hdf files.
    dir_path+file_prefix+fileNum_str+'.hdf5

    Parameters
    ----------
    dir_path : str
        Path to directory of hdf5 files.
    file_prefix : str
        Prefix of hdf5 files.
    total_files : int
        Number of hdf5 files: total_files must be an integer.
    files_per_hdf: int
            Number of files merged in one hdf5 file.
    filetype : str
        Type of file name notation. Fill merged number of files with zeros or X. If not specified, go through all subfolders and create filelist with all hdf5 files.

    Returns
    -------
    filelist : list
        List of paths to hdf5 files.
    '''

    filelist = []

    if filetype == 'zeros':
        for fileNum in range(0, total_files, files_per_hdf):
            fileNum_str = str(fileNum).zfill(6)
            filelist.append(dir_path+file_prefix+fileNum_str+'.hdf5')

    elif filetype == 'X' or filetype == 'x':
        # folder structure: /data/user/shaefs/bundle10/CORSIKA/20904/level3_dev/0000000-0000999 .../0001000
        # with file names e.g. Level2_IC86.2020_corsika.020904.00474X.hdf5 
        FILES_PER_FOLDER = 1000
        hdf_files_total = total_files // files_per_hdf

        for iter_num in range(0, hdf_files_total, FILES_PER_FOLDER):
            _subfolder = str(iter_num).zfill(7) + '-' + str(iter_num+FILES_PER_FOLDER-1).zfill(7)

            # iterate over files in subfolder
            for fileNum in range(0, FILES_PER_FOLDER):

                num_x = len(str(files_per_hdf)) - 1
                _num_merged = (fileNum+iter_num)
                fileNum_str = str(_num_merged).zfill(6-num_x) + 'X'*num_x

                filelist.append(dir_path+_subfolder+'/'+file_prefix+fileNum_str+'.hdf5')
                
    else:
        # create paths to all files listed in dir_path with os, go also in subfolders
        print('Create filelist going through all subfolders.')
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".hdf5"):
                    filelist.append(os.path.join(root, file))
        

    if len(filelist) != total_files//files_per_hdf:
        print(f'Warning: Number of files in generated filelist ({len(filelist)}) should be equal to given num_files/files_per_hdf ({total_files//files_per_hdf}).')
    return filelist

def index_resample_data(weights, test_size, rng):
    '''
    Resample data according to weights.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Weights of each event.
    test_size : int
        Number of events to resample.

    Returns
    -------
    inds : numpy.ndarray
        Indices of resampled events.
    '''

    test_sample_size = len(weights)
    weights_test_normed = weights/np.sum(weights)
    inds = rng.choice(
            np.arange(test_sample_size),
            p=weights_test_normed,
            replace=True,
            size=test_size
    )
    return inds


# ---------- general ----------

def create_logbins(log_E_min, log_E_max, delta_logE, overflow_bins=True):
    '''
    Create logarithmic energy bins between log_E_min and log_E_max with delta_logE.
    Underflow and overflow bins are added.
    If necessary E_max is reduced to fit the bin width.

    Parameters
    ----------
    log_E_min : float
        Minimum log10(E).
    log_E_max : float
        Maximum log10(E).
    delta_logE : float
        Logarithmic bin width.
    overflow_bins : bool
        Add under and overflow bins.
        
    Returns
    -------
    target_bins : numpy.ndarray
        Array of energy bins.
    '''

    n_bins = int((log_E_max - log_E_min) / delta_logE)
    target_bins = np.logspace(log_E_min, log_E_max, n_bins+1)
    print(f'{n_bins} bins between {log_E_min} and {log_E_max} with delta_logE={(log_E_max - log_E_min) / n_bins}.')

    # add under and overflow bin
    if overflow_bins:
        target_bins = np.insert(target_bins, 0, 0)
        target_bins = np.insert(target_bins, n_bins+2, 10**30)
    
    print('Target energy bins:\n', target_bins)
    return target_bins

def get_weights_shiftgamma(weights, primary_energy, delta_gamma: float):
    '''
    Calculate new weights with shifted spectral index by delta_gamma.

    Parameters
    ----------
    weights : array-like
        Weights of each event.
    primary_energy : array-like
        Primary energy of each event.
    delta_gamma : float
        Shift of spectral index.

    Returns
    -------
    w_new : array-like
        New weights.
    '''
    # check if shape is equal
    if len(weights) != len(primary_energy):
        raise ValueError('Length of weights and primary_energy must be equal.')
    
    w_new = weights * (primary_energy)**delta_gamma

    return w_new

# calc livetime
def calc_livetime(run_nr, livetime_i):
    '''
    Calculate the total livetime in seconds.

    Parameters:
    -----------
    run_nr : pd.Series
        Run number.

    livetime_i : pd.Series
        Livetime per run.
    '''

    # get unique run numbers
    run_nr_unique = run_nr.unique()
    
    # live time per run, only count first row of each run
    livetime = 0
    for run in run_nr_unique:
        livetime += livetime_i[run_nr == run].iloc[0]

    return livetime