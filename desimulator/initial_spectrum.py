import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

model_index =pd.read_csv('../models/model_index.csv')
flag = ~np.isnan(model_index['halpha_sfr']) & ~np.isnan(model_index['stellar_mass']) & (model_index['stellar_mass']>10.75)
model_index = model_index[flag]

print('Loaded Model Index \n' , model_index)

def shape_model_index(feature_list, model_index=model_index):

    model_features = [[model_index[feature].tolist()[i] for feature in feature_list] for i in range(len(model_index))]

    return model_features


def find_sfh_neighbors(match_features, feature_list, n_neighbors=5, model_index=model_index):
    
    model_features = shape_model_index(feature_list)
    neighbor_finder = NearestNeighbors(n_neighbors=n_neighbors)
    neighbor_finder.fit(model_features)

    neighbors = neighbor_finder.kneighbors(match_features)

    neighbors_df = model_index.iloc[neighbors[1][0].tolist()]

    return neighbors_df

def find_neighbor_spectra(neighbors_df, wavelength_file='../models/wavelengths',
                          spectra_dir='../models/model_spectra/'):

    wl = np.genfromtxt(wavelength_file)

    spectra_list = [np.genfromtxt(spectra_dir + neighbors_df['file'].tolist()[i] + '.mod').transpose()[1] 
                    for i in range(len(neighbors_df))]
    spectra_list_stellar = [np.genfromtxt(spectra_dir + neighbors_df['file'].tolist()[i] + '.mod').transpose()[2] 
                            for i in range(len(neighbors_df))]
    
    mean_spectrum = np.mean(spectra_list, axis=0)
    mean_spectrum_stellar = np.mean(spectra_list_stellar, axis=0)

    np.savetxt('initial_spectrum.mod', np.array([wl, mean_spectrum, mean_spectrum_stellar]).transpose(),
               header='wl    full model    stellar_model', fmt='%0.2f')

    return wl, mean_spectrum, mean_spectrum_stellar


if __name__ == '__main__':

    time_step, sfr = np.genfromtxt('../sample_data/SFH_tot_SF10_wind0.txt').transpose()
    initial_sfr = np.log10(np.mean(sfr[(time_step > 295) & (time_step < 305)]))

    neighbors_df = find_sfh_neighbors([11, initial_sfr], ['stellar_mass', 'halpha_sfr'])
