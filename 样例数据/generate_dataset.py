import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser(
    description="《交通大数据：理论与方法》样例数据生成。")
parser.add_argument('-d', '--data', nargs='?', default=os.getcwd(),
                    type=str, help="数据存储路径。")
parser.add_argument('-v', '--version', action='version', 
                    version='1.0', help="版本信息。")
_args = parser.parse_args()

data_path = Path(_args.data)

def extract_features(date):
    name = 'A'

    if not os.path.exists(f'{name}_ori_2016{date}.csv'):
        raw_gps = data_path / f'gps_2016{date}'
        if not os.path.exists(raw_gps):
            print(f"未找到{raw_gps.as_posix()}")
            return
        df = pd.read_csv(
            raw_gps, names=['driverID', 'orderID', 'timestamp', 'lon', 'lat'])

        """
        数据压缩，选择研究区域数据，同时删除错误数据
        如经纬度明显异常的数据
        """
        left_geo = 104.081083   #左边界
        right_geo = 104.111122  #右边界
        up_geo = 30.674219      #上边界
        down_geo = 30.653965    #下边界

        df_1 = df[(df.lon>left_geo) & (df.lon<right_geo)
                & (df.lat>down_geo) & (df.lat<up_geo)]
        df_1.to_csv('%s_ori_2016%d.csv' %(name, date), index=None) #存储为文件

        del df
        df = df_1
    else:
        df = pd.read_csv('%s_ori_2016%d.csv' % (name, date))

    if not os.path.exists(f'{name}-速度加速度计算-{date}.csv'):
        """
        去除空值
        """
        print('去除前数据条数：', len(df))
        df = df.dropna()
        print('去除后数据条数：', len(df))

        """
        去除重复数据
        """
        print('去除前数据条数：', len(df))
        df = df.drop_duplicates()
        print('去除后数据条数：', len(df))

        """
        坐标系转换
        """
        from osgeo import osr

        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)  #wgs-84坐标系
        inp = osr.SpatialReference()
        inp.ImportFromEPSG(3857)    #Pseudo-Mercator坐标系
        # 定义坐标转换
        transformation = osr.CoordinateTransformation(wgs84, inp)

        #转换坐标
        xy = df[['lon', 'lat']].apply(
            lambda x: transformation.TransformPoint(x[1], x[0])[:2], axis=1)

        # xy为一个list，每一个元素为一个tuple
        # 转换为dataframe中的两列
        df['x'] = [x[0] for x in xy]
        df['y'] = [x[1] for x in xy]

        """
        离散化
        """
        # 转换为utc+8时区
        df.timestamp = df.timestamp + 8 * 3600
        # currDay为当日0时的timestamp
        currDay = pd.Timestamp('2016%d' % date).timestamp()
        df['time_id'] = (df.timestamp.values - currDay) // 600

        # 空间网格计算
        left = df.x.min()   # 网格左边界
        down = df.y.min()   # 网格下边界
        unit = 50           # 定义网格大小
        df['rowid'] = (df.y.values - down) // unit
        df['colid'] = (df.x.values - left) // unit

        """
        交通流参数计算
        """
        df = df.sort_values(
            by=['driverID', 'orderID', 'timestamp']).reset_index(drop=True)
        # 将订单id，下移一行，用于判断相邻记录是否属于同一订单
        df['orderFlag'] = df['orderID'].shift(1)
        df['identi'] = (df['orderFlag']==df['orderID'])
        # 将坐标、时间戳下移一行，从而匹配相邻轨迹点
        df['x1'] = df['x'].shift(1)
        df['y1'] = df['y'].shift(1)
        df['timestamp1'] = df['timestamp'].shift(1)
        # 将不属于同一订单的轨迹点对删去
        df = df[df['identi']==True]

        # 计算距离
        dist = np.sqrt(np.square(
            (df['x'].values-df['x1'].values)) + np.square(
                (df['y'].values-df['y1'].values)))
        # 计算时间
        time = df['timestamp'].values - df['timestamp1'].values
        # 计算速度
        df['speed'] = dist / time
        # 删去无用列
        df = df.drop(columns=['x1', 'y1', 'orderFlag', 'timestamp1', 'identi'])

        # 计算加速度
        df['speed1'] = df.speed.shift(1)
        df['timestamp1'] = df.timestamp.shift(1)
        df['identi'] = df.orderID.shift(1)
        df = df[df.orderID==df.identi]

        df.loc[:, 'acc'] = (df.speed1.values - df.speed.values) / (
            df.timestamp1.values - df.timestamp.values)
        df = df.drop(columns=['speed1', 'timestamp1', 'identi'])

        df.to_csv(f'{name}-速度加速度计算-{date}.csv', index=None)
    else:
        df = pd.read_csv(f'{name}-速度加速度计算-{date}.csv')

    if not os.path.exists('featureNew-%d.csv' % date):
        """
        计算网格交通流参数
        """
        orderGrouped = df.groupby(['rowid', 'colid', 'orderID', 'time_id'])
        gridGrouped = df.groupby(['rowid', 'colid', 'time_id'])

        # 网格平均车速
        grouped_speed = orderGrouped.speed.mean().reset_index()
        grouped_speed = grouped_speed.groupby(['rowid', 'colid', 'time_id'])
        grid_speed = grouped_speed.speed.mean()
        grid_speed = grid_speed.clip(
            grid_speed.quantile(0.05), grid_speed.quantile(0.95))

        # 网格平均加速度
        grid_acc = gridGrouped.acc.mean()

        # 网格流量
        grouped_volume = orderGrouped.speed.last().reset_index()
        grouped_volume = grouped_volume.groupby(['rowid', 'colid', 'time_id'])
        grid_volume = grouped_volume['speed'].size()
        grid_volume = grid_volume.clip(
            grid_volume.quantile(0.05), grid_volume.quantile(0.95))

        # 网格车速标准差
        grid_v_std = gridGrouped.speed.std()

        # 网格平均停车次数
        stopNum = gridGrouped.speed.agg(lambda x: (x==0).sum())
        grid_stop = pd.concat((stopNum, grid_volume), axis=1)
        grid_stop['stopNum'] = stopNum.values / grid_volume.values
        grid_stop = grid_stop['stopNum']
        grid_stop = grid_stop.clip(0, grid_stop.quantile(0.95))

        # 网格最小车速
        t = orderGrouped['timestamp'].last() - orderGrouped['timestamp'].first()
        dist = np.sqrt(
            (orderGrouped['x'].last().values - orderGrouped['x'].first().values)**2 + \
            (orderGrouped['y'].last().values - orderGrouped['y'].first().values) ** 2)
        grid_min_speed = t.reset_index()
        grid_min_speed['minSpeed'] = dist / t.values
        grid_min_speed = grid_min_speed.groupby(
            ['rowid', 'colid', 'time_id']).minSpeed.mean()

        # 网格自由流车速
        grid_free_speed = df.groupby(
            ['rowid', 'colid'], as_index=False).speed.max()
        grid_free_speed.columns = ['rowid', 'colid', 'freeSpeed']

        feature = pd.concat([
            grid_speed, grid_acc, grid_volume, grid_v_std, 
            grid_stop, grid_min_speed], axis=1).reset_index()
        feature.columns = [
            'rowid', 'colid', 'time_id', 'aveSpeed', 
            'gridAcc', 'volume', 'speed_std', 'stopNum', 'minSpeed']
        feature = pd.merge(
            feature, grid_free_speed, how='left', on=['rowid', 'colid'])
        feature['speedRatio'] = feature.minSpeed / feature.freeSpeed
        feature['minSpeed'].fillna(0, inplace=True)
        feature['speedRatio'].fillna(0, inplace=True)
        feature.to_csv('featureNew-%d.csv' % date, index=None)
    

def do_kmeans(n_clusters, data):
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=0, algorithm='auto').fit(data)
    return kmeans.labels_  


def cluster_ana(cluster_method='kmeans'):
    data_ori = pd.read_csv('train-v20.csv')
    data_ori = data_ori.dropna()

    scaler = MinMaxScaler()
    scaler.fit(data_ori[['aveSpeed', 'gridAcc', 'speed_std', 'stopNum']])
    data_ori_nor = scaler.transform(
        data_ori[['aveSpeed', 'gridAcc', 'speed_std', 'stopNum']])
    data_ori_nor = pd.DataFrame(
        data_ori_nor, columns=['aveSpeed', 'gridAcc', 'speed_std', 'stopNum'])
    data = data_ori_nor.values
    
    methods = {'kmeans': do_kmeans}
    n_clusters = 3
    labels = methods[cluster_method](n_clusters, data)
    
    data_ori['labels'] = labels
    data_ori.to_csv('DATASET-B.csv', index=None)
    
    
def generate_labels():
    jar = []
    for dt in range(1101, 1131):
        feature = 'featureNew-%d.csv' % dt
        if os.path.exists(feature):
            df = pd.read_csv(feature)
            df['date'] = '2016' + str(dt)
            jar.append(df)
        else:
            print(f"未找到'{feature}'。")
            
    df = pd.concat(jar, axis=0)
    df.to_csv('train-v20.csv', index=None)
    cluster_ana()
    
    
if __name__ == '__main__':
    for date in range(1101, 1131):
        extract_features(date)
    generate_labels()
    print("DATASET-B生成完毕！")
    