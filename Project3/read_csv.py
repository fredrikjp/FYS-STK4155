from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload


@dataclass
class DataGroup: 
    """ Object with data containing feature info with respect to a uniqe set of 
    voiprocessing_parameters, preprocessing_parameters and feature """
    data_group: pd.DataFrame = field(repr=False) # Group of data wiTrueth same parameter combinations that is to be analysised
    series_names: np.ndarray
    voi_names: np.ndarray

    feature_name: str = field(init=False)
    voiprocessing_parameters: str = field(init=False)
    preprocessing_parameters: str = field(init=False)

    name_matrix: np.ndarray = field(init=False, repr=False) # Matirx with series + voi_name
    feature_matrix: np.ndarray = field(init=False, repr=False) # Matrix with feature values ordered as in name_matrix
    volume_matrix: np.ndarray = field(init=False, repr=False) # Matrix with volumes --||--

    def __post_init__(self): 
        self.__set_feature_name()
        self.__set_voiprocessing_parameters()
        self.__set_preprocessing_parameters()
        self.__set_data_matrix()
        
    def __set_feature_name(self):
        feature_name = self.data_group['feature_name'].unique()
        assert(len(feature_name) == 1)
        self.feature_name = feature_name[0]


    def __set_voiprocessing_parameters(self):
        voiprocessing_parameters = self.data_group['voiprocessing_parameters'].unique()
        assert(len(voiprocessing_parameters) == 1)
        self.voiprocessing_parameters = voiprocessing_parameters[0]

    def __set_preprocessing_parameters(self):
        preprocessing_parameters = self.data_group['preprocessing_parameters'].unique()
        assert(len(preprocessing_parameters) == 1)
        self.preprocessing_parameters = preprocessing_parameters[0]

    def __set_data_matrix(self):
        """ Creates feature_matrix, volume_matrix"""
        feature_matrix = np.zeros(shape=(len(self.voi_names), len(self.series_names)))
        volume_matrix = np.zeros_like(feature_matrix)
        name_matrix = np.empty_like(feature_matrix, dtype=object)
        for j, voi in enumerate(self.voi_names):
            for i, series in enumerate(self.series_names): 
                result = self.data_group.query('series == @series and name == @voi')
                feature_matrix[j, i] = result['value']
                volume_matrix[j, i] = result['ref1']
                name_matrix[j, i] = ' '.join((series, voi))

        self.name_matrix = name_matrix
        self.feature_matrix = feature_matrix
        self.volume_matrix = volume_matrix

@dataclass(frozen=True)
class ReadCSV:
    """ Class used to read and manipulate CSV data with features """

    input_path: str # Path to csv file that is anlysed
    # logger       # Print to log file
    output_path: str = '../../Data/analyse_tmp.csv' # TODO

    group_by: list = field(default_factory = lambda: ['voiprocessing_parameters', 'preprocessing_parameters', 'feature_name']) # Organize csv data with resepct to groups

    csv_data: pd.DataFrame = field(init=False, repr=False) 
    data_objects: list = field(init=False, repr=False)  # List with data frames containing csv data orderd in groups
    series_list: list = field(init=False)
    voi_list: list = field(init=False)
    n_features: int = field(init=False)

    # Dataframes with feature and target values
    X: pd.DataFrame() = field(init=False, default=None)
    y: pd.DataFrame() = field(init=False, default=None)
     


    def __post_init__(self):
        self.__read_csv_data()


    def __read_csv_data(self):
        """
        Features is gruped by voiprocessing_parameters, preprocessing_parameters and feature_name
            NOT: feature_parameters in csv. 
        """
        # data = pd.read_csv(self.input_path)
        data = pd.read_csv(self.input_path)

        # Get info from full dataset
        series_names = data['series'].unique()
        voi_names = data['name'].unique()
        self.__create_object_groups(data, series_names, voi_names)

    def __create_object_groups(self, data: pd.DataFrame, series_names: np.ndarray, voi_names: np.ndarray):
        # Create Group objects
        gb = data.groupby(self.group_by)
        data_objects = [ DataGroup(gb.get_group(g), series_names, voi_names) for g in gb.groups ] 
        object.__setattr__(self, 'n_features', len(data_objects))
        object.__setattr__(self, 'csv_data', data)
        object.__setattr__(self, 'data_objects', data_objects)
        object.__setattr__(self, 'series_list', series_names)
        object.__setattr__(self, 'voi_list', voi_names)

    #  __  __           _ _  __         ____        _        
    # |  \/  | ___   __| (_)/ _|_   _  |  _ \  __ _| |_ __ _ 
    # | |\/| |/ _ \ / _` | | |_| | | | | | | |/ _` | __/ _` |
    # | |  | | (_) | (_| | |  _| |_| | | |_| | (_| | || (_| |
    # |_|  |_|\___/ \__,_|_|_|  \__, | |____/ \__,_|\__\__,_|
    #                           |___/                        

    def _sort_series(self, index_list: list): 
        assert(len(index_list) == len(self.series_list))
        assert(np.max(index_list) == len(index_list)-1)
        assert(len(np.unique(index_list)) == len(index_list))
        
        old_series_list = self.series_list
        new_series_list = np.empty_like(old_series_list, dtype=object)
        for old, new in enumerate(index_list): 
            new_series_list[new] = old_series_list[old]
        self.__create_object_groups(self.csv_data, new_series_list, self.voi_list)

    def _remove_series(self, index):
        """
        Parameters:
            index: int or array of ints - index with respect to series that is to be removed. 
        """
        old_series_list = self.series_list
        new_series_list = np.delete(old_series_list, index)
        self.__create_object_groups(self.csv_data, new_series_list, self.voi_list)


    def get_group(self, index):
        try:
            return self.data_objects[index]
        except IndexError:
            raise IndexError(f'Index range: [0, {len(self.data_objects)}]')

    def create_df(self, categorical=True) -> None:
        """
        Creates instance variables X and y
        """
        X = pd.DataFrame()
        for i in range(self.n_features): 
            feature = self.get_group(i)
            name = feature.feature_name
            X[name] = np.ravel(feature.feature_matrix, 'F')

        targets = []
        n_vois = len(feature.voi_names)
        for i, series in enumerate(feature.series_names): 
            targets.extend([i]*n_vois)

        y = np.ndarray(targets)
        if categorical == True: 
            y = pd.Categorical.from_codes(targets, feature.series_names)
            y = pd.get_dummies(y)

        object.__setattr__(self, 'X', X)
        object.__setattr__(self, 'y', y)

    def get_df(self, categorical: bool = True) -> tuple: 
        """
        """
        if isinstance(self.X, type(None)) and isinstance(self.y, type(None)): 
            self.create_df(categorical)
            
        return self.X, self.y

    # def get_columns(self): 
    #     """
    #     Returns:
    #         Datframe coloumns of X and y
    #     """
    #     if isinstance(self.X, type(None)) and isinstance(self.y, type(None)): 
    #         self.create_df(categorical)

    #     return self.X.coloumns, self.y.coloumns


    def list_groups(self): 
        for i in range(len(self.data_objects)):
            print(f'==================== \
                  \nindex: {i} \
                  \nfeature: {self.data_objects[i].feature_name} \
                  \nvoiprocessing_parameters: {self.data_objects[i].voiprocessing_parameters} \
                  \npreprocessing_parameters: {self.data_objects[i].preprocessing_parameters} \
                  ')
