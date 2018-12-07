import tensorflow as tf
from ConvReccurrentCells import ConvReccurrentCell


def convrecurrent2d(channels,num_input, time_step,filter_channels, kernel_x,name,kind,batch_size,x=None,initial_state=None,skip_conection=False):

  with tf.variable_scope(name_or_scope=name,reuse=tf.AUTO_REUSE):
      convlstm = ConvReccurrentCell(

          conv_ndims=2,
          input_shape=[time_step, num_input, channels],
          output_channels=filter_channels,
          kernel_shape=[kernel_x,1],
          use_bias=True,
          kind=kind,
          t_max=time_step,
          skip_connection=skip_conection)
      if initial_state == None:
          hidden = convlstm.zero_state(batch_size, tf.float32)
      else:
          hidden = initial_state

      if x == None:
          x = tf.zeros([batch_size,time_step, num_input, filter_channels],tf.float32)
      else:
          x = tf.convert_to_tensor(x)
      y_1, hidden = convlstm(x, hidden)
  return y_1, hidden

def conv2d(x,W,b,strides=1):
  x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
  x = tf.nn.bias_add(x,b)
  return x



def ConvRecurrent(x,arq,filter_channels,kind, num_inputs,time_steps,batch_size,kernel, B_kernel,WF,BF,Wout,Bout):


  conv0 = conv2d(x, kernel, B_kernel, strides=1)
  convlstm0,init0 = convrecurrent2d(channels=filter_channels,num_input=num_inputs, time_step=time_steps,filter_channels=filter_channels,
                         kernel_x=4,name='convlstm0',x=conv0,kind=kind,batch_size=batch_size)
  if arq==1:
      conv0 = conv2d(x, kernel, B_kernel, strides=1)
      convlstm0, init0 = convrecurrent2d(channels=filter_channels, num_input=num_inputs, time_step=time_steps,
                                         filter_channels=filter_channels,
                                         kernel_x=4, name='convlstm0', x=conv0, kind=kind,batch_size=batch_size)
      convlstm2, _ = convrecurrent2d(channels=filter_channels, num_input=num_inputs, time_step=time_steps,
                                     filter_channels=filter_channels,
                                     kernel_x=4, name='convlstm2', initial_state=init0, kind=kind,batch_size=batch_size)
      fc1 = tf.reshape(convlstm2, [batch_size, filter_channels * time_steps * num_inputs])
  elif arq==0:
      conv0 = conv2d(x, kernel, B_kernel, strides=1)
      convlstm0, init0 = convrecurrent2d(channels=filter_channels, num_input=num_inputs, time_step=time_steps,
                                         filter_channels=filter_channels,
                                         kernel_x=4, name='convlstm0', x=conv0, kind=kind,batch_size=batch_size)
      fc1 = tf.reshape(convlstm0, [batch_size, filter_channels * time_steps * num_inputs])

  fc1 = tf.add(tf.matmul(fc1, WF), BF)
  fc1 = tf.tanh(fc1/3)
  out1 = tf.add(tf.matmul(fc1, Wout), Bout)

  return  tf.abs(out1)

def Scoring(Y_true, Y_pred):
  h = Y_pred - Y_true
  g = (-(h-tf.abs(h))/2.0)
  f = ((tf.abs(h)+h)/2.0)
  return tf.reduce_sum(tf.exp(g/13.0)-tf.ones(h.shape))+tf.reduce_sum(tf.exp(f/10.0)-tf.ones(h.shape))


def RMSE(Y_true, Y_pred):
  return tf.sqrt(tf.reduce_sum(tf.square(Y_pred -Y_true))/len(Y_true))

