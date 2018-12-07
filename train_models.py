import tensorflow as tf
from get_data_CMAPSS import *
from model import *
import time
import numpy as np



class model:
    def __init__(self,X_train,Y_train,
                 X_cross,Y_cross,
                 X_test,Y_test,
                 X_test_b, Y_test_b,
                 model_path,
                 learning_rate,training_epoch,
                 batch_size_train,batch_size_test,batch_size_test_b,
                 display_step,
                 num_inputs,time_steps,
                 num_hidden,num_outputs,
                 filter_channels,arq,
                 kind,name,save=False):
        self._X_train = X_train
        self._Y_train = Y_train
        self._X_cross = X_cross
        self._Y_cross = Y_cross
        self._X_test = X_test
        self._Y_test = Y_test
        self._X_test_b = X_test_b
        self._Y_test_b = Y_test_b
        self._model_path = model_path
        self._learning_rate = learning_rate
        self._training_epoch = training_epoch
        self._batch_size_train = batch_size_train
        self._batch_size_test = batch_size_test
        self._batch_size_test_b = batch_size_test_b
        self._display_step = display_step
        self._num_inputs = num_inputs
        self._time_steps = time_steps
        self._num_hidden = num_hidden
        self._num_outputs = num_outputs
        self._filter_channels = filter_channels
        self._arq = arq
        self._kind = kind
        self._name = name
        self._save = save

    def train(self):
        tf.reset_default_graph()

        kernel = tf.get_variable("kernel_c", shape=[15, 1, 1, self._filter_channels],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        B_kernel = tf.get_variable("B_kernel", shape=[self._filter_channels],
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d())

        WF = tf.get_variable('WF', shape=[self._filter_channels * self._time_steps * self._num_inputs, self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())
        BF = tf.get_variable('BF', shape=[self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())

        Wout = tf.get_variable('Wout', shape=[self._num_hidden, self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())
        Bout = tf.get_variable('Bout', shape=[self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())

        X = tf.placeholder("float", [self._batch_size_train, self._time_steps, self._num_inputs, 1])
        Y = tf.placeholder("float", [self._batch_size_train, self._num_outputs])

        pred1 = ConvRecurrent(X, self._arq, self._filter_channels, self._kind,
                              self._num_inputs, self._time_steps, self._batch_size_train,
                              kernel, B_kernel, WF,
                              BF, Wout, Bout)
        # score = Scoring(Y_true=Y_test,Y_pred=pred)
        # rmse = RMSE(Y_true=Y_test, Y_pred=pred)

        loss_op = tf.reduce_sum(tf.square(Y - pred1)) + tf.reduce_sum(tf.abs(WF)) + \
                  100 * tf.reduce_sum(tf.square(tf.nn.relu(pred1 - 125)))

        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        # optimize for training
        train_op = optimizer.minimize(loss_op)

        # evaluate model
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(Y - pred1)))

        # initializate variables
        init = tf.global_variables_initializer()
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        print("training...")

        with tf.Session() as sess:
            sess.run(init)
            # saver.restore(sess, model_path)
            start = time.time()

            for step in range(1, self._training_epoch + 1):

                batch_x, batch_y = Next_Batch3(self._X_train, self._Y_train, self._batch_size_train)
                batch_x = batch_x.reshape((self._batch_size_train, self._time_steps, self._num_inputs, 1))
                batch_x_crossval, batch_y_crossval = Next_Batch3(self._X_cross, self._Y_cross, self._batch_size_train)
                batch_x_crossval = batch_x_crossval.reshape((self._batch_size_train, self._time_steps, self._num_inputs, 1))

                # pred = sess.run(accuracy, feed_dict={X: batch_x_crossval, Y: batch_y_crossval, keep_prob: 1.0})
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

                if self._save == True:
                    if step % self._display_step == 0 or step == 1:
                        # Calculate batch loss and accuracy
                        _, acc_train = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                        print("\n" + "Step " + str(step))
                        _, acc_cross = sess.run([loss_op, accuracy],
                                                     feed_dict={X: batch_x_crossval, Y: batch_y_crossval})
                    
                        if self._arq==0:
                            acc_file1 = self._model_path + "Conv" + str(self._kind)+"/RMSE_hist/ACC_train.txt"
                            acc_file2 = self._model_path + "Conv" + str(self._kind) + "/RMSE_hist/ACC_cross.txt"

                        else:
                            acc_file1 = self._model_path + "Conv" + str(self._kind) + "_ED/RMSE_hist/ACC_train.txt"
                            acc_file2 = self._model_path + "Conv" + str(self._kind)+"_ED/RMSE_hist/ACC_cross.txt"
                        file1 = open(acc_file1, "a+")
                        file2 = open(acc_file2, "a+")
                        file1.write("{:.4f}".format(acc_train) + "\n")
                        file2.write("{:.4f}".format(acc_cross) + "\n")
                        file1.close()
                        file2.close()
                    if step == self._training_epoch:
                        batch_x, batch_y = Next_Batch3(self._X_train, self._Y_train, self._batch_size_train)
                        batch_x = batch_x.reshape((self._batch_size_train, self._time_steps, self._num_inputs, 1))
                        pred = sess.run(pred1, feed_dict={X: batch_x, Y: batch_y})
                        if self._arq == 0:
                            predicciones = self._model_path + "Conv" + str(self._kind) + "/ajuste.txt"

                        else:
                            predicciones = self._model_path + "Conv" + str(self._kind) + "_ED/ajuste.txt"

                        ajuste =  np.column_stack((batch_y, pred))
                        np.savetxt(predicciones, ajuste, delimiter=" ")

                    



            print("Optimization Finished!")
            if self._arq == 0:
                        eltime = self._model_path + "Conv" + str(self._kind) + "/time.txt"

            else:
                eltime = self._model_path + "Conv" + str(self._kind) + "_ED/time.txt"

            stop = time.time()
            t = stop - start
            file3 = open(eltime, "a+")
            file3.write("{:.4f}".format(t) + "\n")
            file3.close()
            print('\n Tiempo total de entrenamiento = %g [s] \n' % t)
            # Save model weights to disk

            save_path = saver.save(sess, self._model_path)
            print("Model saved in file: %s" % save_path)

    def test(self):
        tf.reset_default_graph()

        kernel = tf.get_variable("kernel_c", shape=[15, 1, 1, self._filter_channels],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        B_kernel = tf.get_variable("B_kernel", shape=[self._filter_channels],
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d())

        WF = tf.get_variable('WF', shape=[self._filter_channels * self._time_steps * self._num_inputs, self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())
        BF = tf.get_variable('BF', shape=[self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())

        Wout = tf.get_variable('Wout', shape=[self._num_hidden, self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())
        Bout = tf.get_variable('Bout', shape=[self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())

        X = tf.placeholder("float", [self._batch_size_test, self._time_steps, self._num_inputs, 1])
        Y = tf.placeholder("float", [self._batch_size_test, self._num_outputs])

        pred1 = ConvRecurrent(X, self._arq, self._filter_channels, self._kind,
                              self._num_inputs, self._time_steps, self._batch_size_test,
                              kernel, B_kernel, WF,
                              BF, Wout, Bout)

        # rmse = RMSE(Y_true=Y_test, Y_pred=pred)

        loss_op = tf.reduce_sum(tf.square(Y - pred1)) + tf.reduce_sum(tf.abs(WF)) + \
                  100 * tf.reduce_sum(tf.square(tf.nn.relu(pred1 - 125)))

        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        # optimize for training
        train_op = optimizer.minimize(loss_op)

        # evaluate model
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(Y - pred1)))
        score = Scoring(Y_true=Y, Y_pred=pred1)


        # initializate variables
        init = tf.global_variables_initializer()
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        print("testing all data ...")

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self._model_path)
            self._X_test = self._X_test.reshape((self._batch_size_test, self._time_steps, self._num_inputs, 1))
            #start = time.time()
            #pred = sess.run(pred1, feed_dict={X: self._X_test, Y: self._Y_test})
            #print("Testing all data Accuracy:", \
            #      sess.run(accuracy, feed_dict={X: self._X_test, Y: self._Y_test}))
            #stop = time.time()
            #t = stop - start
            #print('\n Tiempo total de testing = %g [s] \n' % t)
            #print("Testing all data Score:", \
            #      sess.run(score, feed_dict={X: self._X_test, Y: self._Y_test}))



    def test_b(self):
        tf.reset_default_graph()

        kernel = tf.get_variable("kernel_c", shape=[15, 1, 1, self._filter_channels],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        B_kernel = tf.get_variable("B_kernel", shape=[self._filter_channels],
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d())

        WF = tf.get_variable('WF', shape=[self._filter_channels * self._time_steps * self._num_inputs, self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())
        BF = tf.get_variable('BF', shape=[self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())

        Wout = tf.get_variable('Wout', shape=[self._num_hidden, self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())
        Bout = tf.get_variable('Bout', shape=[self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())

        X = tf.placeholder("float", [self._batch_size_test_b, self._time_steps, self._num_inputs, 1])
        Y = tf.placeholder("float", [self._batch_size_test_b, self._num_outputs])

        pred1 = ConvRecurrent(X, self._arq, self._filter_channels, self._kind,
                              self._num_inputs, self._time_steps, self._batch_size_test_b,
                              kernel, B_kernel, WF,
                              BF, Wout, Bout)
        
        # rmse = RMSE(Y_true=Y_test, Y_pred=pred)

        loss_op = tf.reduce_sum(tf.square(Y - pred1)) + tf.reduce_sum(tf.abs(WF)) + \
                  100 * tf.reduce_sum(tf.square(tf.nn.relu(pred1 - 125)))

        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        # optimize for training
        train_op = optimizer.minimize(loss_op)

        # evaluate model
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(Y - pred1)))
        score = Scoring(Y_true=Y, Y_pred=pred1)

        # initializate variables
        init = tf.global_variables_initializer()
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        print("testing...")

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self._model_path)
            self._X_test_b = self._X_test_b.reshape((self._batch_size_test_b, self._time_steps, self._num_inputs, 1))
            #start = time.time()
            #pred = sess.run(pred1, feed_dict={X: self._X_test, Y: self._Y_test})
            #print("Testing Accuracy:", \
            #      sess.run(accuracy, feed_dict={X: self._X_test_b, Y: self._Y_test_b}))
            #stop = time.time()
            #t = stop - start
            #print('\n Tiempo total de testing = %g [s] \n' % t)
            #print("Testing Score:", \
            #      sess.run(score, feed_dict={X: self._X_test_b, Y: self._Y_test_b}))
            pred = sess.run(pred1, feed_dict={X: self._X_test_b, Y: self._Y_test_b})
            if self._arq == 0:
                predicciones = self._model_path + "Conv" + str(self._kind) + "/predicciones.txt"
                scort = self._model_path + "Conv" + str(self._kind) + "/score.txt"
                rms = self._model_path + "Conv" + str(self._kind) + "/RMSE.txt"


            else:
                predicciones = self._model_path + "Conv" + str(self._kind) + "_ED/predicciones.txt"
                scort = self._model_path + "Conv" + str(self._kind) + "_ED/score.txt"
                rms = self._model_path + "Conv" + str(self._kind) + "_ED/RMSE.txt"

            
            file1 = open(scort, "a+")
            file2 = open(rms, "a+")
             
            file1.write("{:.4f}".format(sess.run(score, feed_dict={X: self._X_test_b, Y: self._Y_test_b})) + "\n")
            file2.write("{:.4f}".format(sess.run(accuracy, feed_dict={X: self._X_test_b, Y: self._Y_test_b})) + "\n")
            rmse =  np.column_stack((self._Y_test_b, pred))
            np.savetxt(predicciones, rmse, delimiter=" ")
            
            file1.close()
            file2.close()

    def cross(self):
        tf.reset_default_graph()

        kernel = tf.get_variable("kernel_c", shape=[15, 1, 1, self._filter_channels],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        B_kernel = tf.get_variable("B_kernel", shape=[self._filter_channels],
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d())

        WF = tf.get_variable('WF', shape=[self._filter_channels * self._time_steps * self._num_inputs, self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())
        BF = tf.get_variable('BF', shape=[self._num_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())

        Wout = tf.get_variable('Wout', shape=[self._num_hidden, self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())
        Bout = tf.get_variable('Bout', shape=[self._num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())

        X = tf.placeholder("float", [2048, self._time_steps, self._num_inputs, 1])
        Y = tf.placeholder("float", [2048, self._num_outputs])#2048 batch size cross

        pred1 = ConvRecurrent(X, self._arq, self._filter_channels, self._kind,
                              self._num_inputs, self._time_steps, 2048,
                              kernel, B_kernel, WF,
                              BF, Wout, Bout)
        # score = Scoring(Y_true=Y_test,Y_pred=pred)
        # rmse = RMSE(Y_true=Y_test, Y_pred=pred)

        loss_op = tf.reduce_sum(tf.square(Y - pred1)) + tf.reduce_sum(tf.abs(WF)) + \
                  100 * tf.reduce_sum(tf.square(tf.nn.relu(pred1 - 125)))

        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        # optimize for training
        train_op = optimizer.minimize(loss_op)

        # evaluate model
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(Y - pred1)))

        # initializate variables
        init = tf.global_variables_initializer()
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        print("testing...")

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self._model_path)
            batch_x_crossval, batch_y_crossval = Next_Batch3(self._X_cross, self._Y_cross, 2048)
            batch_x_crossval = batch_x_crossval.reshape((2048, self._time_steps, self._num_inputs, 1))
            start = time.time()
            #pred = sess.run(pred1, feed_dict={X: self._X_test, Y: self._Y_test})
            print("Testing Accuracy:", \
                  sess.run(accuracy, feed_dict={X: batch_x_crossval, Y: batch_y_crossval}))
            stop = time.time()
            t = stop - start
            print('\n Tiempo total de testing cross= %g [s] \n' % t)





