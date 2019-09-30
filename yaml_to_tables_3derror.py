import yaml
import pandas as pd
import numpy as np
import scipy.stats as stats

class_names = ('ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
               'iron', 'lamp', 'phone')

bias_data_mean = {}
bias_data_median = {}
bias_data_mode = {}


for cls in class_names:
    bias_data_mean[cls] = {}
    bias_data_median[cls] = {}
    bias_data_mode[cls] = {}

    file_name = '3derror_logs/' + cls + '_3derror_logs.yml'
    outfile = open(file_name, 'r')
    data = yaml.load(outfile)
    df = pd.DataFrame.from_dict(data, orient='index')

    bias_data_mean[cls]['x'] = float(df.mean()['x'])
    bias_data_mean[cls]['y'] = float(df.mean()['y'])
    bias_data_mean[cls]['z'] = float(df.mean()['z'])

    bias_data_median[cls]['x'] = float(df.median()['x'])
    bias_data_median[cls]['y'] = float(df.median()['y'])
    bias_data_median[cls]['z'] = float(df.median()['z'])

    bias_data_mode[cls]['x'] = float(df.mode()['x'][0])
    bias_data_mode[cls]['y'] = float(df.mode()['y'][0])
    bias_data_mode[cls]['z'] = float(df.mode()['z'][0])

    x = sorted(list(df['x']))  #sorted
    y = sorted(list(df['y']))  #sorted
    z = sorted(list(df['z']))  #sorted

    fit_x = stats.norm.pdf(x, np.mean(x), np.std(x))  #this is a fitting indeed
    fit_y = stats.norm.pdf(y, np.mean(y), np.std(y))  #this is a fitting indeed
    fit_z = stats.norm.pdf(z, np.mean(z), np.std(z))  #this is a fitting indeed

with open('cfgs/bias_linemod_mean.yml', 'w') as outfile:
    yaml.dump(bias_data_mean, outfile, default_flow_style=False)

with open('cfgs/bias_linemod_median.yml', 'w') as outfile:
    yaml.dump(bias_data_median, outfile, default_flow_style=False)

with open('cfgs/bias_linemod_mode.yml', 'w') as outfile:
    yaml.dump(bias_data_mode, outfile, default_flow_style=False)