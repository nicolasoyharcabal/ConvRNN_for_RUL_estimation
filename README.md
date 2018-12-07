# ConvRNN for RUL estimation in mechanical systems

Code used in Thesis: 
"Convolutional Recurrent Neurtal Networks for Remaining Useful Life Prediction in Mechanical Systems"
N.Oyharcabal, Bachelor Thesis, University of Chile, 2018.

# Abstract
The determination of the Remaining Useful Life (RUL) of a machine, equipment, device
or mechanical element, is a crucial issue for the future of the industry and the optimiza-
tion of processes as in the case of maintenance. The continuous monitoring of machines
along with a good prediction of the RUL, allows the minimization of maintenance costs
and lower exposure to catastrophic faults. On the other hand, it is also known that data
obtained from monitoring is varied, has a sequential nature and do not always has a
strict relationship with RUL, so their estimation is difficult problem.
Nowadays in this class of problems, different kinds of Neural Networks are used. In par-
ticular when it is wanted to model problems with sequential data, Recurrent Neural Net-
work (RNN) are preferred for its capacity to autonomously identify patterns in temporal
sequences and recently, there are also alternatives that incorporate the Convolution as
an operation in each cell of these Networks. Therefore these Networks in some cases are
better than their convolutional and recurrent pairs, since they are capable of processing
sequences of images, and in the particular case of this work, time series of monitoring
data that are softened by convolution and processed by recurrence.
The general objective of this work is to obtain the best alternative based on Convolu-
tional Recurrent Neural Network (ConvRNN) for determining the RUL of a turbofan from
the time series of C-MAPSS dataset. It is also studied how to modify the database to im-
prove the accuracy of a ConvRNN and the application of Convolution as a primary oper-
ation in a time series whose parameters show the behavior of a turbofan. For this, a Con-
volutional LSTM, Convolutional LSTM Encoder-Decoder, Convolutional JANET and Con-
volutional JANET Encoder-Decoder are programmed. Then, it is proven that the model
that obtains the best results in terms of average accuracy and number of parameters
necessary (less is better because less memory is needed) for the Network is the Convo-
lutional JANET Encoder-Decoder that is also able to successfully assimilate the totality of
the C-MAPSS databases. Moreover, it is also found as in other works, that the RUL from
the database can be modified for data before failure.
For the start-up of Neural Networks, computers from the SRMI laboratory of the Me-
chanical Engineering Department of the University of Chile are used.

# Requirements

Python 3.5,
Tensorflow 1.4,
Numpy,
Scikit-learn,
Pandas.
