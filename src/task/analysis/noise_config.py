from itertools import product

import pandas as pd

codes = ['290.1', '250.2', '714.0|714.1', '428.2', '411.2']

noise_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

config = list(product(codes, noise_level))

# config.append(['290.1', 0.0])
# config.append(['290.1', 0.3])

df_config = pd.DataFrame(config, columns=['phenotypes', 'noise_levels'])

df_config.to_csv('/home/vauvelle/pycharm-sftp/pheprob/src/jobs/configs/pheno_noise-{}.csv'.format(df_config.shape[0]),
                 index=False)
