import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""
Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation
https://ti.arc.nasa.gov/publications/154/download/

Turbofan Engine Degradation Simulation Data Set
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ 
"""

def TrainImages(n_set,n_flights ,filter_t,f_modes):
  # Data should be a pandas dataframe of the original raw data, e.g. FD001.txt
  # RUL is a pandas col with the corresponding RUL value of every time step
  # n_flights is the number of flights found for this dataset e.g. FD001 has 100 flights
  images = []
  rul = []
  
  set = '/media/inti/BC27-082F/C-MAPSS/train_FD00'+str(n_set)+'.txt'
  data = pd.read_csv(set, delim_whitespace=True, header=None)
  vuelos = np.linspace(1, n_flights, n_flights)
  data.columns = ['Unit Number', 'Cycles', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3',
                    'Sensor Measurement  1', 'Sensor Measurement  2', 'Sensor Measurement  3', 'Sensor Measurement  4',
                    'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  7', 'Sensor Measurement  8',
                    'Sensor Measurement  9', 'Sensor Measurement  10', 'Sensor Measurement  11',
                    'Sensor Measurement  12', 'Sensor Measurement  13', 'Sensor Measurement  14',
                    'Sensor Measurement  15', 'Sensor Measurement  16', 'Sensor Measurement  17',
                    'Sensor Measurement  18', 'Sensor Measurement  19', 'Sensor Measurement  20',
                    'Sensor Measurement  21']
  data['RUL'] = pd.Series(np.zeros(len(data)), index=data.index)
  for vuelo in vuelos:
      a = data[data['Unit Number'] == vuelo]
      tiempo = len(a)
      RUL = np.linspace(tiempo, 0, tiempo)
      for n in range(0, len(RUL)):
          if RUL[n] > 125:#RUL edition for 99.99% reliability
              RUL[n] = 125
      data.loc[a.index, 'RUL'] = pd.Series(RUL, index=a.index)

  for i in range(1, n_flights + 1):  # Iterate over each flight in the dataset
      flight = data[data['Unit Number'] == i]  # Take all data entries for the current flight
      RUL = flight['RUL'].to_frame()  # Extract the RUL labels contained for this
      if f_modes:
          X = flight.drop(
              ['Unit Number',  # 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3',
               'Sensor Measurement  1', 'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  10',
               'Sensor Measurement  16', 'Sensor Measurement  18', 'Sensor Measurement  19', 'RUL'], axis=1)
      else:
          X = flight.drop(
              ['Unit Number', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3',
               'Sensor Measurement  1', 'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  10',
               'Sensor Measurement  16', 'Sensor Measurement  18', 'Sensor Measurement  19', 'RUL'], axis=1)

      length = len(X)
      if filter_t > length:
          continue

      n_images = length - filter_t + 1  # Num of possible images for each flight
      X = np.array(X)
      RUL = np.array(RUL)
      for j in range(n_images):
          image = X[j:j + filter_t, :]
          images.append(image)
          rul.append(RUL[j + filter_t-1])

  images = np.asarray(images)
  images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
  rul = np.asarray(rul)


  return images, rul

def TestImages(n_set,n_flights ,filter_t,f_modes):
  # Obtain only the last image of each flight in the dataset
  # In this case, we do not extract the RUL, since is given in a separate file
  images = []
  set_images = '/media/inti/BC27-082F/C-MAPSS/test_FD00'+str(n_set)+'.txt'
  set_rul = '/media/inti/BC27-082F/C-MAPSS/RUL_FD00'+str(n_set)+'.txt'
  Test_data = pd.read_csv(set_images, delim_whitespace=True, header=None)
  Test_data.columns = ['Unit Number', 'Cycles', 'Operational Setting 1', 'Operational Setting 2',
                       'Operational Setting 3',
                       'Sensor Measurement  1', 'Sensor Measurement  2', 'Sensor Measurement  3',
                       'Sensor Measurement  4',
                       'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  7',
                       'Sensor Measurement  8',
                       'Sensor Measurement  9', 'Sensor Measurement  10', 'Sensor Measurement  11',
                       'Sensor Measurement  12',
                       'Sensor Measurement  13', 'Sensor Measurement  14', 'Sensor Measurement  15',
                       'Sensor Measurement  16',
                       'Sensor Measurement  17', 'Sensor Measurement  18', 'Sensor Measurement  19',
                       'Sensor Measurement  20',
                       'Sensor Measurement  21']

  for i in range(1, n_flights + 1):
      flight = Test_data[Test_data['Unit Number'] == i]
      if f_modes:
          X = flight.drop(
              ['Unit Number',  # 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3',
               'Sensor Measurement  1', 'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  10',
               'Sensor Measurement  16', 'Sensor Measurement  18', 'Sensor Measurement  19'], axis=1)
      else:
          X = flight.drop(
              ['Unit Number', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3',
               'Sensor Measurement  1', 'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  10',
               'Sensor Measurement  16', 'Sensor Measurement  18', 'Sensor Measurement  19'], axis=1)

      length = len(X)
      if filter_t > length:
          continue

      X = np.array(X)
      image = X[(length - filter_t):length, :]
      images.append(image)

  images = np.asarray(images)
  
  images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
  rul = np.array(pd.read_csv(set_rul,delim_whitespace=True,header=None)) #RUL labels for the test set

  if n_flights == 100:
      n_units=100
  elif n_flights == 248:
      n_units = 248
  else:
      n_units = 259
  for i in range(n_units):
      if rul[i] > 125:
          rul[i] = 125

  return images,rul


def get_data(window,f_modes=True,data1=False):
    if f_modes:
        X1, Y1 = TrainImages(1, 100, window,f_modes=f_modes)
        X_test1, Y_test1 = TestImages(1, 100, window,f_modes=f_modes)
        X2, Y2 = TrainImages(2, 259, window,f_modes=f_modes)
        X_test2, Y_test2 = TestImages(2, 259, window,f_modes=f_modes)

        X1, X_crossval1, Y1, Y_crossval1 = train_test_split(X1, Y1, test_size=0.15, random_state=48)
        X2, X_crossval2, Y2, Y_crossval2 = train_test_split(X2, Y2, test_size=0.15, random_state=48)

        X = np.concatenate([X1, X2], 0)
        Y = np.concatenate([Y1, Y2], 0)
        X_crossval = np.concatenate([X_crossval1, X_crossval2], 0)
        Y_crossval = np.concatenate([Y_crossval1, Y_crossval2], 0)
        X_test = np.concatenate([X_test1, X_test2], 0)
        Y_test = np.concatenate([Y_test1, Y_test2], 0)

        X3, Y3 = TrainImages(3, 100, window,f_modes=f_modes)
        X_test3, Y_test3 = TestImages(3, 100, window,f_modes=f_modes)

        X3, X_crossval3, Y3, Y_crossval3 = train_test_split(X3, Y3, test_size=0.15, random_state=48)

        X = np.concatenate([X, X3], 0)
        Y = np.concatenate([Y, Y3], 0)
        X_crossval = np.concatenate([X_crossval, X_crossval3], 0)
        Y_crossval = np.concatenate([Y_crossval, Y_crossval3], 0)
        X_test = np.concatenate([X_test, X_test3], 0)
        Y_test = np.concatenate([Y_test, Y_test3], 0)

        X4, Y4 = TrainImages(4, 248, window,f_modes=f_modes)
        X_test4, Y_test4 = TestImages(4, 248, window,f_modes=f_modes)

        X4, X_crossval4, Y4, Y_crossval4 = train_test_split(X4, Y4, test_size=0.15, random_state=48)

        X_train = np.concatenate([X, X4], 0)
        Y_train = np.concatenate([Y, Y4], 0)
        X_crossval = np.concatenate([X_crossval, X_crossval4], 0)
        Y_crossval = np.concatenate([Y_crossval, Y_crossval4], 0)
        X_test = np.concatenate([X_test, X_test4], 0)
        Y_test = np.concatenate([Y_test, Y_test4], 0)

        sc_X = MinMaxScaler(feature_range=(-1, 1))
        X_train = sc_X.fit_transform(X_train)  # Only transform the trainning set
        X_crossval = sc_X.transform(X_crossval)
        X_test = sc_X.transform(X_test)
        X_test2 = sc_X.transform(X_test2)
        X_test4 = sc_X.transform(X_test4)

        return X_train, Y_train, X_crossval, Y_crossval, X_test, Y_test, X_test2, Y_test2, X_test4, Y_test4
    if f_modes==False and data1==True:
        X1, Y1 = TrainImages(1, 100, window,f_modes=f_modes)
        X_test1, Y_test1 = TestImages(1, 100, window,f_modes=f_modes)

        X_train1, X_crossval1, Y_train1, Y_crossval1 = train_test_split(X1, Y1, test_size=0.15, random_state=48)

        sc_X = MinMaxScaler(feature_range=(-1, 1))
        X_train1 = sc_X.fit_transform(X_train1)  # Only transform the trainning set
        X_crossval1 = sc_X.transform(X_crossval1)
        X_test1 = sc_X.transform(X_test1)

        return X_train1, Y_train1, X_crossval1, Y_crossval1, X_test1, Y_test1

   


    else:
        X1, Y1 = TrainImages(1, 100, window,f_modes=f_modes)
        X_test1, Y_test1 = TestImages(1, 100, window,f_modes=f_modes)
        X3, Y3 = TrainImages(3, 100, window,f_modes=f_modes)
        X_test3, Y_test3 = TestImages(3, 100, window,f_modes=f_modes)

        X1, X_crossval1, Y1, Y_crossval1 = train_test_split(X1, Y1, test_size=0.15, random_state=48)
        X3, X_crossval3, Y3, Y_crossval3 = train_test_split(X3, Y3, test_size=0.15, random_state=48)

        X_train = np.concatenate([X1, X3], 0)
        Y_train = np.concatenate([Y1, Y3], 0)
        X_crossval = np.concatenate([X_crossval1, X_crossval3], 0)
        Y_crossval = np.concatenate([Y_crossval1, Y_crossval3], 0)
        X_test = np.concatenate([X_test1, X_test3], 0)
        Y_test = np.concatenate([Y_test1, Y_test3], 0)

        sc_X = MinMaxScaler(feature_range=(-1, 1))
        X_train = sc_X.fit_transform(X_train)  # Only transform the trainning set
        X_crossval = sc_X.transform(X_crossval)
        X_test = sc_X.transform(X_test)
        X_test1 = sc_X.transform(X_test1)
        X_test3 = sc_X.transform(X_test3)

        return X_train, Y_train, X_crossval, Y_crossval, X_test, Y_test, X_test1, Y_test1, X_test3, Y_test3


def Next_Batch3(X,y,batch_size):
    batch_x=[]
    batch_y=[]
    for j in range(1, batch_size + 1):
        i = np.random.randint(1, len(X))
        image = X[i, :]
        rul = y[i]
        batch_x.append(image)
        batch_y.append(rul)
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)
    return batch_x, batch_y
