import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import seaborn as sns
from toolbox.wololo import redshift2lumdistance
from toolbox import plot_tools

starlight_catalog_dir = 'C:\\Users\\ariel\\Workspace\\Organized PhD Catalog\\'
starlight_sample = Table.read(starlight_catalog_dir + 'sample_flagged_lowz.fits')
starlight_sample_plus = Table.read(starlight_catalog_dir + 'sample_plus_flagged_lowz.fits')
starlight_results = Table.read(starlight_catalog_dir + 'catalog_PHO_flagged_lowz.fits')

time_step, sfr = np.genfromtxt('../sample_data/SFH_tot_SF10_wind0.txt').transpose()
initial_sfr = np.log10(np.mean(sfr[(time_step > 295) & (time_step < 305)]))

sf_flag = np.log10(starlight_sample_plus['nii_6584_flux']/starlight_sample_plus['halpha_flux']) < -0.4

redshift = starlight_sample['z']
halpha_flux = starlight_sample_plus['halpha_flux'] * 1e-17
distance = redshift2lumdistance(redshift, unit='cm')
halpha_luminosity = halpha_flux * 4 * np.pi * distance ** 2
halpha_sfr = np.log10(halpha_luminosity / (10 ** 41.28))

model_index = pd.DataFrame({'file': starlight_results['file'].astype('str')[sf_flag],
                            'stellar_mass': starlight_sample_plus['mcor_gal'][sf_flag].tolist(),
                            'atmass': starlight_results['atmass'][sf_flag].tolist(),
                            'atflux': starlight_results['atflux'][sf_flag].tolist(),
                            'halpha_sfr': halpha_sfr[sf_flag]})

print(model_index)
model_index.to_csv('model_index.csv')


sns.set_style('ticks')

mass_flag = (model_index['stellar_mass'] > 6.5) & (model_index['stellar_mass'] < 12.1)

fig = plt.figure(figsize=(10,7.5))
plt.hexbin(model_index['stellar_mass'][mass_flag], model_index['halpha_sfr'][mass_flag],
           C=model_index['atflux'][mass_flag], cmap='Spectral_r', gridsize=30, mincnt=5,
           edgecolors='w', linewidths=0.1)
# sns.kdeplot(x=model_index['stellar_mass'], y=model_index['halpha_sfr'], color='k')
plt.scatter(11, initial_sfr, s=100, color='k', edgecolor='w')
plt.xlabel(r'$\log\,M_\star/M_\odot$', fontsize=20)
plt.ylabel(r'$\log\, \mathrm{SFR \, [M_\odot/yr]}$', fontsize=20)
sns.despine()
# fig.tight_layout()
plt.savefig('C:\\Users\\ariel\\find_galaxy.png', dpi=300)
plt.show()
