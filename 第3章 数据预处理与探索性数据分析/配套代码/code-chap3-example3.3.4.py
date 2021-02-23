import pandas as pd
data = pd.read_csv('DATASET-B.csv')
data_vol = data[data['time_id']==50].groupby(['date'])['volume'].sum()
df = pd.DataFrame(data_vol)
df = df.reset_index(drop=False)
from datetime import datetime
df['day'] = df['date'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d").weekday()+1)
# weekday()函数返回值：周一为0，周日为6