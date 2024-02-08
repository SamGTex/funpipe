import pandas as pd
import numpy as np
import simweights

class DataManager:
    def __init__(self):
        self.df_raw = pd.DataFrame()
        self.event_type = False
        self.target = None

    def read_to_df(self, filelist, path_feature_names, total_files, model):
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

        for file in filelist:
            print(f'Read in file {file}')
            tmp_df = pd.HDFStore(file, "r")
            if weighter is None:
                weighter = simweights.CorsikaWeighter(tmp_df,total_files)
            else:
                weighter += simweights.CorsikaWeighter(tmp_df,total_files)
        print()

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

        print('Done.\n')

        return self.df_raw

    def __eventtype_label__(self, he_muon, etot_bundle, Qmax, Qtot, Esep_value, Qsep_value):
        # definition of leading, bundle and balloon event
        mask_bundle = (he_muon/etot_bundle < Esep_value) #0
        mask_leading = (he_muon/etot_bundle > Esep_value) & (Qmax/Qtot < Qsep_value) #1
        mask_balloon = (he_muon/etot_bundle > Esep_value) & (Qmax/Qtot > Qsep_value) #2

        y = np.zeros_like(he_muon)
        y[mask_bundle] = 0
        y[mask_leading] = 1
        y[mask_balloon] = 2

        return y
    
    def create_eventtype(self, name_Eleading, nameEbundle, nameQmax, nameQtot, cut_E, cut_Q):
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
        self.df_raw['event_type'] = self.__eventtype_label__(self.df_raw[name_Eleading], self.df_raw[nameEbundle], self.df_raw[nameQmax], self.df_raw[nameQtot], cut_E, cut_Q)
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

    def apply_cuts(self, conditions, inplace=True):
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

        if inplace:
            self.df_raw = self.df_raw[mask]
            self.df_raw.reset_index(drop=True, inplace=True)
            return self.df_raw

        else:
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
def create_path_list(dir_path, file_prefix, total_files, files_per_hdf, filenum_old=True):
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
        Number of hdf5 files: total_files/1000 must be an integer.
    files_per_hdf: int
            Number of files merged in one hdf5 file.
    filenum_old : bool
        If True, use old file numbering, using zeros to fill up to merged files numbers
        If False, use new file numbering, using X to fill up to merged files numbers
            
    Returns
    -------
    filelist : list
        List of paths to hdf5 files.
    '''

    filelist = []

    if filenum_old:
        for fileNum in range(0, total_files, files_per_hdf):
            fileNum_str = str(fileNum).zfill(6)
            filelist.append(dir_path+file_prefix+fileNum_str+'.hdf5')
    else:
        # loop over subfolders with 1000 files
        for _runnum_lower in range(0, total_files, 1000):
            _runnum_upper = _runnum_lower + 999

            #_subfolder = str(_runnum_lower).zfill(7) + '-' + str(_runnum_upper).zfill(7)
            _subfolder =''
            # loop over files in subfolder
            # e.g. Level2_IC86.2020_corsika.020904.0161XX.i3.bz2, XX because of 100 files per hdf5
            # for 1000 files per hdf5: 016XXX
            # get number of x
            for file_num in range(_runnum_lower, _runnum_upper, files_per_hdf):
                num_x = len(str(files_per_hdf)) - 1

                # substitute last num_x digits of file_num with num_x times X
                _filename = str(file_num).zfill(6)
                _filename = _filename[:-num_x] + 'X'*num_x

                filelist.append(dir_path+_subfolder+file_prefix+_filename+'.hdf5')

    return filelist

def index_resample_data(weights, test_size):
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
    inds = np.random.choice(
            np.arange(test_sample_size),
            p=weights_test_normed,
            replace=True,
            size=test_size
    )
    return inds

def create_logbins(log_E_min, log_E_max, delta_logE):
    n_bins = int((log_E_max - log_E_min) / delta_logE)
    target_bins = np.logspace(log_E_min, log_E_max, n_bins+1)
    print(f'{n_bins} bins between {log_E_min} and {log_E_max} with delta_logE={delta_logE}')

    # add under and overflow bin
    target_bins = np.insert(target_bins, 0, 0)
    target_bins = np.insert(target_bins, n_bins+2, 10**30)
    
    print('Target energy bins:\n', target_bins)
    return target_bins