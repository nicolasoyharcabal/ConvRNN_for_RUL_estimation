import tensorflow as tf
from train_models import *


learning_rate = 0.001

batch_size = 1024
display_step = 100


# Network Parameters

num_hidden = 100# num hidden units
num_outputs = 1 # Only one output
veces = 20


name = "Model"
model_path = "/media/inti/BC27-082F/LSTM_PRUEBA_FINAL/PRUEBA_FINAL"




test = {"FD001":[15,30,100,False,2000,True],
        "FD002":[18,21,259,True,30000,False],
        "FD003":[15,30,100,False,20000,False],
        "FD004":[18,19,248,True,30000,False]}#[num_inputs,time_steps,test_size,f_modes]
modelo = {"JANET":[[0,2],[1,10]],
          "LSTM": [[0, 2], [1, 10]]}#arquetipo,filter_chanels



for i in modelo:
    
    for j in range(0,2):
        arq = modelo[i][j][0]
        
        filter_channels = modelo[i][j][1]

        for k in range(1,5):
            a = "FD00" + str(k)
            
            model_path = "/media/inti/BC27-082F/LSTM_PRUEBA_FINAL/PRUEBA_FINAL/" + a +"/"

            if k== 1 :
                for h in range(veces):
                    a = "FD00" + str(k)
                    if (h+1) == veces:
                        save=True
                    else:
                        save=False
                    
                    X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a = get_data(
                        window=test[a][1], f_modes=test[a][3],data1=True)
                    modelx = model(X_train, Y_train,
                                   X_crossval, Y_crossval,
                                   X_test_a, Y_test_a,
                                   X_test_a, Y_test_a,
                                   model_path,
                                   learning_rate, test[a][4],
                                   1024, X_test_a.shape[0], test[a][2],
                                   display_step,
                                   test[a][0], test[a][1],
                                   num_hidden, num_outputs,
                                   filter_channels, arq,
                                   kind=i, name=name,save=save)
                    modelx.train()
                    modelx.test_b()
                    


            elif k==2:
                for h in range(veces):
                    X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a, X_test_b1, Y_test_b1, X_test_b2, Y_test_b2 = get_data(
                        window=test[a][1], f_modes=test[a][3])
                    modelx = model(X_train, Y_train,
                                   X_crossval, Y_crossval,
                                   X_test_a, Y_test_a,
                                   X_test_b1, Y_test_b1,
                                   model_path,
                                   learning_rate, test[a][4],
                                   1024, X_test_a.shape[0], test[a][2],
                                   display_step,
                                   test[a][0], test[a][1],
                                   num_hidden, num_outputs,
                                   filter_channels, arq,
                                   kind=i, name=name)
                    modelx.train()
                    modelx.test_b()
            elif k==3:
                
                for h in range(veces):
                    a = "FD00" + str(k)
                    
                    X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a, X_test_b1, Y_test_b1, X_test_b2, Y_test_b2 = get_data(
                        window=test[a][1], f_modes=test[a][3])
                    modelx = model(X_train, Y_train,
                                   X_crossval, Y_crossval,
                                   X_test_a, Y_test_a,
                                   X_test_b2, Y_test_b2,
                                   model_path,
                                   learning_rate, test[a][4],
                                   1024, X_test_a.shape[0], test[a][2],
                                   display_step,
                                   test[a][0], test[a][1],
                                   num_hidden, num_outputs,
                                   filter_channels, arq,
                                   kind=i, name=name)
                    modelx.train()
                    modelx.test_b()
            else:
                for h in range(veces):
                    X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a, X_test_b1, Y_test_b1, X_test_b2, Y_test_b2 = get_data(
                        window=test[a][1], f_modes=test[a][3])
                    modelx = model(X_train, Y_train,
                                   X_crossval, Y_crossval,
                                   X_test_a, Y_test_a,
                                   X_test_b2, Y_test_b2,
                                   model_path,
                                   learning_rate, test[a][4],
                                   1024, X_test_a.shape[0], test[a][2],
                                   display_step,
                                   test[a][0], test[a][1],
                                   num_hidden, num_outputs,
                                   filter_channels, arq,
                                   kind=i, name=name)
                    modelx.train()
                    modelx.test_b()






