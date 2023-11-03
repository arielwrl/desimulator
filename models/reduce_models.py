import numpy as np
import os

model_dir = '/home/ariel/Workspace/desimulator/models/'
model_list = os.listdir(model_dir)

print('I found', len(model_list), 'models in', model_dir)

for file_name in model_list:

    print('\nReducing model:', file_name)

    try:
        wl, model_full, model_stellar, model_nebular = np.genfromtxt(model_dir + file_name).transpose()

    except Exception:
        print('Skipped')
        continue

    reduced_flag = (wl > 2500) & (wl < 10000)

    wl = wl[reduced_flag]
    model_full = model_full[reduced_flag]
    model_stellar = model_stellar[reduced_flag]

    np.savetxt(model_dir + file_name, np.array([wl, model_full, model_stellar]).transpose(), fmt='%0.2f',
               header='wl    model_full    model_stellar')

