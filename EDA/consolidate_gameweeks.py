import pandas as pd
import glob
import re

all_files = glob.glob('/Users/vijith.poovelil/Documents/sandbox/Fantasy-Premier-League/data/2023-24/gws/gw*.csv')

li = []

for filename in all_files:
    match = re.search(r'gw(\d+)\.csv', filename)
    if match:
        gw_num = int(match.group(1))
        df = pd.read_csv(filename, index_col=None, header=0)
        df['gw'] = gw_num
        li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.to_csv('/Users/vijith.poovelil/Documents/sandbox/Fantasy-Premier-League/data/2023-24/gws/merged_gw.csv', index=False)