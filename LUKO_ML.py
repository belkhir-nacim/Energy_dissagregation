import gc
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score, accuracy_score


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def scale(x, data_min, data_max, newmin=0, newmax=1):
    data_range = data_max - data_min
    scale_ = ((newmax - newmin) / _handle_zeros_in_scale(data_range))
    min_ = newmin - data_min * scale_
    x = x * scale_
    x += min_
    return x


def xp_ml(datasetfile='data/toy_dataset.csv'):
    """
    :return:
    """
    df = pd.read_csv(datasetfile)
    df['label'].replace("mains", 0, inplace=True)
    df['label'].replace("refrigerator", 1, inplace=True)
    y = np.array(df['label'])
    del df['label']
    X = np.array(df)
    del df

    fold = StratifiedShuffleSplit(y=y, n_iter=15, test_size=0.3)
    i = 1
    try:
        dfml = pd.read_csv('score_ml.csv')
    except:
        dfml = pd.DataFrame()

    for ml in ["svm linear", "rf", "logistic"]:
        perf = []
        perf_jaccard = []
        for train_index, test_index in fold:
            xtrain, ytrain, xtest, ytest = X[train_index], y[train_index], X[test_index], y[test_index]
            print(np.min(xtrain))
            print(np.max(xtrain))
            exit()
            scale = MinMaxScaler()
            xtrain = scale.fit_transform(xtrain)

            if ml == 'svm linear':
                estimator = LinearSVC()
            elif ml == 'rf':
                estimator = RandomForestClassifier(n_estimators=20, n_jobs=4)
            elif ml == 'logistic':
                estimator = LogisticRegression(dual=True, n_jobs=4)
            else:
                raise ValueError('try one each these classifier: [ "svm linear", "rf", "logistic"]')

            estimator.fit(xtrain, ytrain)

            xtest = scale.transform(xtest)
            ypred = estimator.predict(xtest)
            perf.append(f1_score(ytest, ypred))
            perf_jaccard.append(jaccard_similarity_score(ytest, ypred))
        sperf = pd.Series(perf)
        sperf_jac = pd.Series(perf_jaccard)
        dfperf = pd.DataFrame()
        dfperf['f1'] = sperf.T
        dfperf['jaccard'] = sperf.T
        dfperf['algo'] = ml
        dfml = dfml.append(dfperf, ignore_index=True)
    dfml.to_csv('score_ml.csv', index=None)


dict_key = {
    'mains': 0,
    'refrigerator': 1,
    'microwave': 2,
    'dishwaser': 3
}


def xp_largedataset():
    """
    :return:
    """
    df = pd.read_csv('data/redd_largerdataset_train.csv')
    df['label'].replace(dict_key, inplace=True)
    y = np.array(df['label'])
    del df['label']
    X = np.array(df)
    del df

    fold = StratifiedShuffleSplit(y=y, n_iter=15, test_size=0.3)
    try:
        dfml = pd.read_csv('score_ml.csv')
    except:
        dfml = pd.DataFrame()

    dfvalidation = pd.read_csv('data/redd_largerdataset_test.csv')
    dfvalidation['label'].replace(dict_key, inplace=True)
    yval = np.array(dfvalidation['label'])
    del dfvalidation['label']
    xval = np.array(dfvalidation)

    for ml in [
        "rf",
        "svm linear",
    ]:
        perf_test = []
        perf_validation = []
        for train_index, test_index in fold:
            xtrain, ytrain, xtest, ytest = X[train_index], y[train_index], X[test_index], y[test_index]
            minn = np.min(xtrain)
            maxx = np.max(xtrain)
            xtrain = scale(xtrain, minn, maxx, 0, 1)

            if ml == 'svm linear':
                estimator = LinearSVC()
            elif ml == 'rf':
                estimator = RandomForestClassifier(n_estimators=20, n_jobs=4)
            elif ml == 'logistic':
                estimator = LogisticRegression(dual=True, n_jobs=4)
            else:
                raise ValueError('try one each these classifier: [ "svm linear", "rf", "logistic"]')

            estimator.fit(xtrain, ytrain)

            xtest = scale(xtest, minn, maxx, 0, 1)
            ypred = estimator.predict(xtest)
            perf_test.append(accuracy_score(ytest, ypred))

            xval = scale(xval, minn, maxx, 0, 1)
            ypred = estimator.predict(xval)
            perf_validation.append(accuracy_score(yval, ypred))

        sperftest = pd.Series(perf_test)
        sperfval = pd.Series(perf_validation)

        dfperf = pd.DataFrame()
        dfperf['value'] = sperftest.T
        dfperf['set'] = 'test'
        dfperf['algo'] = ml
        dfml = dfml.append(dfperf, ignore_index=True)

        dfperf = pd.DataFrame()
        dfperf['value'] = sperfval.T
        dfperf['set'] = 'validation'
        dfperf['algo'] = ml
        dfml = dfml.append(dfperf, ignore_index=True)

    dfml.to_csv('score_ml_largerdataset.csv', index=None)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution1D, LSTM, GlobalMaxPooling1D
from keras.utils import np_utils
from keras.callbacks import History


class ConvolutionalLSTM(object):
    def __init__(self, input_dim,
                 input_length,
                 nb_classes,
                 hidden_dims,
                 nb_filter=64,
                 filter_length=64,
                 batch_size=64,
                 pool_length=16,
                 nb_epoch=1):
        """

        :return:
        """
        self.history = History()
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.input_dim = input_dim
        self.input_length = input_length

        self.model = Sequential()
        # self.model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', activation='sigmoid',input_dim=input_dim))
        # self.model.add(MaxPooling1D())
        self.model.add(LSTM(input_dim=input_dim, output_dim=hidden_dims))

        self.model.add(Dense(input_dim=hidden_dims, output_dim=nb_classes, activation='sigmoid'))
        self.model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="adam")

    def fit(self, X_train, y_train, X_test, y_test):
        print('start fitting')
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        return self.model.fit(X_train, y_train,
                              batch_size=self.batch_size,
                              nb_epoch=self.nb_epoch,
                              validation_data=(X_test, y_test), callbacks=[self.history])

    def predict(self, X_test, batchsize=32):
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        return self.model.predict_classes(X_test, batch_size=batchsize)


class ConvolutionalNetwork1D(object):
    def __init__(self,
                 input_dim,
                 input_length,
                 nb_classes,
                 hidden_dims,
                 nb_filter=64,
                 filter_length=64,
                 batch_size=64,
                 pool_length=16,
                 nb_epoch=1):
        """
        :return:
        """
        self.history = History()
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.input_dim = input_dim
        self.input_length = input_length

        self.model = Sequential()
        self.model.add(Convolution1D(nb_filter=nb_filter,
                                     filter_length=filter_length,
                                     border_mode='same',
                                     activation='sigmoid',
                                     # input_shape=(1,input_length,1)
                                     input_dim=input_dim,
                                     # input_length=input_length
                                     ))
        self.model.add(GlobalMaxPooling1D())

        self.model.add(Dense(hidden_dims))
        # self.model.add(Dropout(0.2))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(hidden_dims))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(hidden_dims / 2))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(nb_classes, activation='sigmoid'))
        print('compiling')
        self.model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="adam")

    def fit(self, X_train, y_train, X_test, y_test):
        print('start fitting')
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        return self.model.fit(X_train, y_train,
                              batch_size=self.batch_size,
                              nb_epoch=self.nb_epoch,
                              validation_data=(X_test, y_test), callbacks=[self.history])

    def predict(self, X_test, batchsize=32):
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        return self.model.predict_classes(X_test, batch_size=batchsize)


def xp_largedataset_convolutionalNetwork():
    """
    :return:
    """
    df = pd.read_csv('data/redd_largerdataset_train.csv')
    df['label'].replace(dict_key, inplace=True)
    nb_classes = len(df['label'].unique())
    y = np.array(df['label'])

    del df['label']
    X = np.array(df)

    del df
    fold = StratifiedShuffleSplit(y=y, n_iter=1, test_size=0.3)
    try:
        dfml = pd.read_csv('score_larger_dataset_DL.csv')
    except:
        dfml = pd.DataFrame()

    dfvalidation = pd.read_csv('data/redd_largerdataset_test.csv')
    dfvalidation['label'].replace(dict_key, inplace=True)
    yval = np.array(dfvalidation['label'])
    del dfvalidation['label']
    xval = np.array(dfvalidation)

    dfplot_ALLXP = pd.DataFrame()
    for ml in [
        "convlstm",
        # "convnet",
    ]:
        iterindex = 1
        dfplot_runs_dict = dict()
        perf_test = []
        perf_validation = []
        for train_index, test_index in fold:

            dfplot_xp = pd.DataFrame()


            xtrain, ytrain, xtest, ytest = X[train_index], y[train_index], X[test_index], y[test_index]

            ytrain = np_utils.to_categorical(ytrain, nb_classes=nb_classes)
            ytest = np_utils.to_categorical(ytest, nb_classes=nb_classes)

            minn = np.min(xtrain)
            maxx = np.max(xtrain)
            xtrain = scale(xtrain, minn, maxx, 0, 1)

            input_length, input_dim = xtrain.shape

            if ml == 'convnet':
                estimator = ConvolutionalNetwork1D(input_dim=input_dim, input_length=input_length,
                                                   nb_classes=nb_classes, hidden_dims=200, nb_epoch=30)
            elif ml == 'convlstm':
                estimator = ConvolutionalLSTM(input_dim=input_dim, input_length=input_length, nb_classes=nb_classes,
                                              hidden_dims=500, nb_epoch=100)
            else:
                raise ValueError('try one each these classifier: [ "svm linear", "rf", "logistic"]')

            xtest = scale(xtest, minn, maxx, 0, 1)
            estimator.fit(xtrain, ytrain, xtest, ytest)
            val_acc, val_loss, loss, acc = estimator.history.history['val_acc'], estimator.history.history['val_loss'], \
                                           estimator.history.history['loss'], estimator.history.history['acc']


            sns.plt.plot(acc,c='red',label='Train Acc')
            sns.plt.plot(val_acc,c='blue',label='Test Acc')

            sns.plt.plot(loss,c='red',ls='--', label='Train Loss')
            sns.plt.plot(val_loss,c='blue',ls='--',label='Test Loss')
            sns.plt.title('Accuracy and Loss of Training and Test Set')
            sns.plt.xlabel('Num Epochs')
            sns.plt.ylabel('Value')
            sns.plt.legend(loc='best')
            sns.plt.savefig('plot{0}_{1}.png'.format(iterindex,ml))
            sns.plt.close()

            tmp = pd.DataFrame()
            d = pd.DataFrame()
            d['iter{0}'.format(iterindex)] = val_acc
            d['perf'] = 'Val Acc'
            tmp = tmp.append(d, ignore_index=True)

            d = pd.DataFrame()
            d['iter{0}'.format(iterindex)] = val_loss
            d['perf'] = 'Val Loss'
            tmp = tmp.append(d, ignore_index=True)

            d = pd.DataFrame()
            d['iter{0}'.format(iterindex)] = loss
            d['perf'] = 'Loss'
            tmp = tmp.append(d, ignore_index=True)

            d = pd.DataFrame()
            d['iter{0}'.format(iterindex)] = acc
            d['perf'] = 'Acc'
            tmp = tmp.append(d, ignore_index=True)

            dfplot_xp = dfplot_xp.append(tmp,ignore_index=True)

            dfplot_runs_dict[iterindex] = dfplot_xp

            xval = scale(xval, minn, maxx, 0, 1)
            ypred = estimator.predict(xval)
            perf_validation.append(accuracy_score(list(yval), list(ypred)))
            gc.collect()
            iterindex+=1


        print(dfplot_xp)


    dfplot_ALLXP.to_csv('fuck.csv',index=False)
    dfml.to_csv('fuck2.csv', index=None)



# def get_data():
#     df = pd.read_csv('xpruns2.csv')
#     for i,d in df.groupby(''):
#
# get_data()



import os
from sklearn.utils import shuffle
def mkdir(path):
    """
    Check if directories exist else it create all the dirs

    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def _train():

    folder_root = mkdir('data')
    file_listfiles = os.path.join(folder_root, 'all_appliance_file.csv')

    print('make a new train set')
    htrain = [6]
    meter = ['mains', 'refrigerator', 'microwave', 'dishwaser']

    dict = {
        'mains': 1,
        'refrigerator': 2,
        'microwave': 3,
        'dishwaser': 4
    }

    all_file = pd.read_csv(file_listfiles, index_col=0)
    all_file = all_file.loc[(all_file.building.isin(htrain)) & (all_file.type.isin(meter))]
    all_file.drop_duplicates('type',inplace=True)
    print(all_file)
    df = pd.DataFrame()
    for i, dataset_info in all_file.groupby('file'):
        ifile = dataset_info.file.unique()[0]
        appliancetype = dataset_info.type.unique()[0]
        print(ifile)
        tmpdf = pd.read_csv(ifile, index_col=0, skiprows=1, header=None)
        tmpdf = tmpdf.T.fillna(tmpdf.min(axis=1)).T
        tmpdf = tmpdf.iloc[range(24)]
        tmpdf['label'] = appliancetype
        df = df.append(tmpdf, ignore_index=True)
    df = shuffle(shuffle(df))

    df_copy = df.copy()
    for col in df_copy.columns:
        df_copy[col] = df_copy['label']

    del df['label']
    del df_copy['label']
    df_copy.replace(dict,inplace=True)
    df = df.values.flatten()
    df_copy = df_copy.values.flatten()

    DF = pd.DataFrame()
    DF['value'] = df
    DF['label'] = df_copy
    print(DF)
    DF.to_csv('allbuilding6.csv')

# _train()
# exit()

def slidingWindow(sequence, winSize, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) / step) + 1

    # Do the work
    for i in range(0, int(numOfChunks * step), step):
        yield sequence[i:i + winSize]





def valid():
    print('read file')
    df = pd.read_csv('data/redd_largerdataset_train.csv')
    df1 = pd.read_csv('data/redd_largerdataset_test.csv')
    df = df.append(df1, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    df.replace(dict_key, inplace=True)

    y = df['label']
    del df['label']
    x = np.array(df)

    minn = np.min(x)
    maxx = np.max(x)
    x = scale(x,minn,maxx,0,1)
    print(' Learn')
    estimator = RandomForestClassifier(n_estimators=50, n_jobs=4)
    #
    estimator.fit(x,y)

    validdf = pd.read_csv('allbuilding6.csv')
    chunk = slidingWindow(range(validdf.shape[0]),3600,3600)
    # i = 1
    # for c in chunk:
    #     print(i)
    #     i+=1
    # exit()
    xp = pd.DataFrame()
    print('start validation')
    a = 1
    for c in chunk:
        print(a)
        a+=1
        val = validdf.iloc[c].copy()
        xtest = list(val['value'])
        ytest = val['label'].unique()[0]
        y_pred = estimator.predict(scale(np.array(xtest).reshape(1,-1),minn,maxx))
        print(y_pred==ytest,end=', ')
        val['predlabel'] = y_pred[0]
        xp = xp.append(val,ignore_index=True)
    xp.to_csv('results2.csv')

# valid()








if __name__ == '__main__':
    #     # xp_ml(datasetfile='data/raw_dataset.csv')
    #     # xp_largedataset()
    xp_largedataset_convolutionalNetwork()

