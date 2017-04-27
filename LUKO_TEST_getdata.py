import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import minmax_scale
def mkdir(path):
    """
    Check if directories exist else it create all the dirs

    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


folder_root = mkdir('data')
file_types = os.path.join(folder_root, 'all_appliances.csv')
file_listfiles = os.path.join(folder_root, 'all_appliance_file.csv')
file_dataset = os.path.join(folder_root,'raw_dataset.csv')

def get_all_redd_data():
    # Parse all building
    list_of_appliances = []
    df_all_file = pd.DataFrame()
    for building in range(1, 7):
        print('building ',building)
        filelabel = '/Users/nacimbelkhir/LUKO/low_freq/house_{b}/labels.dat'.format(b=building)
        dflabel = pd.read_csv(filelabel, header=None, names=['id', 'type'], index_col=None, delim_whitespace=True)

        # Parse All appliances
        for i, label in dflabel.iterrows():
            appliance_id = label.id
            appliance_type = label.type
            list_of_appliances.append(appliance_type)
            folder_appliance = mkdir(os.path.join(folder_root, 'meter_{0}'.format(appliance_type)))
            fileappliance = '/Users/nacimbelkhir/LUKO/redd_low_freq/building{building}/elec/meter{appliance}.csv'.format(
                building=building, appliance=appliance_id)

            df_meter = pd.read_csv(fileappliance, index_col=0, skiprows=3, header=None, names=['W'])

            df_meter.index = pd.to_datetime(df_meter.index, unit='s')
            groupmeters_hourly = df_meter.groupby(pd.TimeGrouper('H'))
            DF = pd.DataFrame()
            for i, df_hourly in groupmeters_hourly:

                tt = df_hourly.resample('S').bfill()
                vals = np.array(tt['W'])
                if len(vals) == 0:
                    pass
                else:
                    DF = DF.append(pd.Series(vals).T, ignore_index=True)

            filetrain = os.path.join(folder_appliance, 'train_b{0}_id{1}.csv'.format(building, appliance_id))
            infofile = pd.Series()
            infofile['type'] = appliance_type
            infofile['building'] = int(building)
            infofile['applianceid'] = int(appliance_id)
            infofile['file'] = filetrain
            df_all_file = df_all_file.append(infofile.T,ignore_index=True)
            DF.to_csv(filetrain)

    d = pd.DataFrame(list_of_appliances, columns=['type'])
    d.drop_duplicates(inplace=True)
    d.reset_index(drop=True, inplace=True)
    d.to_csv(file_types)
    df_all_file.building = df_all_file.building.astype(int)
    df_all_file.applianceid = df_all_file.applianceid.astype(int)
    df_all_file.to_csv(file_listfiles)

def create_dataset_larger():
    """

    :return:
    """
    def _train():
        print('make a new train set')
        htrain = [1,2,3]
        meter= ['mains','refrigerator','microwave','dishwaser']

        dict = {
            'mains':1,
            'refrigerator':2,
            'microwave':3,
            'dishwaser':4
        }

        all_file = pd.read_csv(file_listfiles,index_col=0)
        all_file = all_file.loc[ (all_file.building.isin(htrain)) & (all_file.type.isin(meter))]

        df = pd.DataFrame()
        for i, dataset_info in all_file.groupby('file'):
            ifile = dataset_info.file.unique()[0]
            appliancetype = dataset_info.type.unique()[0]
            print(ifile)
            tmpdf = pd.read_csv(ifile,index_col=0,skiprows=1,header=None)
            tmpdf = tmpdf.T.fillna(tmpdf.min(axis=1)).T
            tmpdf['label'] = appliancetype
            df = df.append(tmpdf,ignore_index=True)
        df.to_csv('data/redd_largerdataset_train.csv',index=None)
    _train()


    def _test():
        print('make a new test set')
        htrain = [4, 5]
        meter = ['mains', 'refrigerator', 'microwave', 'dishwaser']

        dict = {
            'mains': 1,
            'refrigerator': 2,
            'microwave': 3,
            'dishwaser': 4
        }

        all_file = pd.read_csv(file_listfiles, index_col=0)
        all_file = all_file.loc[(all_file.building.isin(htrain)) & (all_file.type.isin(meter))]

        df = pd.DataFrame()
        for i, dataset_info in all_file.groupby('file'):
            ifile = dataset_info.file.unique()[0]
            appliancetype = dataset_info.type.unique()[0]
            print(ifile)
            tmpdf = pd.read_csv(ifile, index_col=0, skiprows=1, header=None)
            tmpdf = tmpdf.T.fillna(tmpdf.min(axis=1)).T
            tmpdf['label'] = appliancetype
            df = df.append(tmpdf, ignore_index=True)
        df.to_csv('data/redd_largerdataset_test.csv', index=None)
    _test()


class Generator(object):
    @classmethod
    def gen_signals_hour(cls, zero=0):
        x = []
        for t in range(1, 3601):
            if (t < 0) or (t > 900):
                x.append(zero)
            else:
                x.append(200 * (1 + np.exp(-t / 10)))
        return x

    @classmethod
    def gen_signature_day(cls, zero=0):
        x = []
        for h in range(24):
            x += cls.gen_signals_hour(zero=zero)
        return x

    @classmethod
    def get_a(cls):
        x = np.zeros(3600 * 24)
        return x

    @classmethod
    def get_b(cls):
        return cls.get_a() + 1000

    @classmethod
    def get_c(cls):
        x = []
        for h in range(24):
            for t in range(1, 3601):
                x.append(1000 * np.cos(2 * np.pi * (t / 3600.)))
        return np.array(x)

    @classmethod
    def get_d(cls):
        x = []
        for h in range(24):
            for t in range(1, 3601):
                x.append(0 + 15 * np.random.rand())
        return np.array(x)

    @classmethod
    def get_e(cls):
        return cls.get_c() + cls.get_d()

    @classmethod
    def get_toy_dataset(cls):
        """
        based on previous data generator we propose to make a dataset of different hourly examples including:
        - signal+signature
        - signal alone
        Let's say for this toy dataset that measures are done on weekdays (5 days) and for 2 weeks= 10 days of 24 hours
        24*10 hourly signal are generated of each aforementionned type of examples.
        if noise== True, then
        """
        df = pd.DataFrame()
        signature = cls.gen_signals_hour(zero=0)
        for i in range(int(24*10)):
            print(i)
            x = cls.get_a()[:3600]
            df = df.append(pd.Series(x).T, ignore_index=True)
            x = cls.get_b()[:3600]
            df = df.append(pd.Series(x).T, ignore_index=True)
            x = cls.get_c()[:3600]
            df = df.append(pd.Series(x).T, ignore_index=True)
            x = cls.get_d()[:3600]
            df = df.append(pd.Series(x).T, ignore_index=True)
            x = cls.get_e()[:3600]
            df = df.append(pd.Series(x).T, ignore_index=True)
        dfsign = df.copy() + signature
        dfsign['label'] = 'refrigerator'
        df['label'] = 'mains'
        df = df.append(dfsign, ignore_index=True)
        df.to_csv('data/toy_dataset.csv', index=None)
        return df

if __name__ == '__main__':
    get_all_redd_data()

    create_dataset_larger()
    # Generator.get_toy_dataset()

