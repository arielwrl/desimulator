import numpy as np
import matplotlib.pyplot as plt
import bagpipes as pipes
import seaborn as sns

def create_bagpipes_model(age_bins_mid, sfrs, Av, logU=-2.5):
    
    square_bursts = []

    dust = {}
    dust["type"] = "Cardelli"
    dust["Av"] = Av

    nebular = {}
    nebular["logU"] = logU

    dust["eta"] = 2

    custom = {}               
    custom["history"] = np.array([age_bins_mid, sfrs]).transpose()
    custom['massformed'] = 8.9
    custom['metallicity'] = 1

    model_components = {}
    
    model_components['custom'] = custom
    model_components["redshift"] = 0.04
    model_components["dust"] = dust  
    model_components["veldisp"] = 75
    model_components["nebular"] = nebular  

    # Wavelength array to plot spectra.
    wl = np.arange(2501, 10000, 1)

    # Creating a model galaxy
    model = pipes.model_galaxy(model_components, spec_wavs=wl)

    return model

    
if __name__ == '__main__':

    age_bins_mid_0, sfrs_0 = np.genfromtxt('../sample_data/SFH_tot_SF10_wind0.txt').transpose()

    age_bins_mid_0 *= 1e6

    model_0 = create_bagpipes_model(age_bins_mid_0, sfrs_0[::-1], Av=0)
    model_0.sfh.plot()
    model_0.plot()

    age_bins_mid_45, sfrs_45 = np.genfromtxt('../sample_data/SFH_tot_SF10_wind45.txt').transpose()

    age_bins_mid_45 *= 1e6

    model_45 = create_bagpipes_model(age_bins_mid_45, sfrs_45[::-1], Av=0)
    model_45.sfh.plot()
    model_45.plot()

    age_bins_mid_90, sfrs_90 = np.genfromtxt('../sample_data/SFH_tot_SF10_wind90.txt').transpose()

    age_bins_mid_90 *= 1e6

    model_90 = create_bagpipes_model(age_bins_mid_90, sfrs_90[::-1], Av=0)
    model_90.sfh.plot()
    model_90.plot()

    age_bins_mid_nowind, sfrs_nowind = np.genfromtxt('../sample_data/SFH_tot_RCSF10.txt').transpose()

    age_bins_mid_nowind *= 1e6

    model_nowind = create_bagpipes_model(age_bins_mid_nowind, sfrs_nowind[::-1], Av=0)
    model_nowind.sfh.plot()
    model_nowind.plot()

    initial_spectrum = np.genfromtxt('../sample_data/initial_spectrum.mod').transpose()
    z = 0.04

    wl = initial_spectrum[0] * (1 + z)
    stellar_model = 1e-17 * initial_spectrum[2] / ( 1+ z)

    sns.set_style('ticks')

    fig = plt.figure(figsize=(10,5))
    plt.plot(wl, model_nowind.spectrum[:,1]+stellar_model, lw=0.5, color='k',
             label='No Wind')
    plt.plot(wl, model_0.spectrum[:,1]+stellar_model, lw=0.5, color='#00F5B8',
             label='Face-on')
    plt.plot(wl, model_45.spectrum[:,1]+stellar_model, lw=0.5, color='#6153CC',
             label='45 degrees')
    plt.plot(wl, model_90.spectrum[:,1]+stellar_model, lw=0.5, color='#A60067',
             label='Edge-on')
    sns.despine()
    plt.ylim(0, 1.5e-15)
    plt.legend(frameon=False,fontsize=20)
    plt.xlabel(r'$\lambda\,\mathrm{[\AA]}$')
    plt.ylabel(r'Flux')
    fig.tight_layout()
    plt.savefig('C:\\Users\\ariel\\final_spectrum.png', dpi=300)
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(wl, stellar_model, lw=0.5, color='k')
    sns.despine()
    plt.xlabel(r'$\lambda\,\mathrm{[\AA]}$')
    plt.ylabel(r'Flux')
    fig.tight_layout()
    plt.savefig('C:\\Users\\ariel\\initial_spectrum.png', dpi=300)