from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Embedding, Bidirectional, LSTM
from keras.layers import Concatenate, Conv1D, BatchNormalization, TimeDistributed, GRU, Reshape
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model


def CNNBasedRNN(MAX_SEQUENCE_LENGTH=None, DIC_SIZE=None, EMBEDDING_DIM=None):
    """ CNN Based RNN in Many to Many""" 

    seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded_input = Embedding(DIC_SIZE, EMBEDDING_DIM, input_length = MAX_SEQUENCE_LENGTH)(seq_input)

    convolution1 = Conv1D(kernel_size=1, filters=32, padding = 'same')(embedded_input)
    convolution2 = Conv1D(kernel_size=2, filters=64, padding = 'same')(embedded_input)
    convolution3 = Conv1D(kernel_size=3, filters=128, padding = 'same')(embedded_input)
    convolution4 = Conv1D(kernel_size=4, filters=256, padding = 'same')(embedded_input)
        
    concatenate = Concatenate(axis=2)([convolution1,convolution2,convolution3,convolution4])
    batchnormalization = BatchNormalization()(concatenate)
    timedistributed1 = TimeDistributed(Dense(300))(batchnormalization)
    timedistributed2 = TimeDistributed(Dense(150))(timedistributed1)
        
    gru = GRU(units=50, return_sequences=True)(timedistributed2)
    timedistributed3 = TimeDistributed(Dense(1))(gru)
    b = Reshape((200,))(timedistributed3)
        
    model = Model(inputs=seq_input, output=b)
    return model

def RNN(MAX_SEQUENCE_LENGTH=None, DIC_SIZE=None, EMBEDDING_DIM=None):
    """ RNN in Many to Many """

    seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded_input = Embedding(DIC_SIZE, EMBEDDING_DIM, input_length = MAX_SEQUENCE_LENGTH, mask_zero=True)(seq_input)

    bilstm = Bidirectional(LSTM(units=100, return_sequences=True), merge_mode='concat')(embedded_input)
    b = TimeDistributed(Dense(3, activation='softmax'))(bilstm)
    #b = Reshape((200,))(timedistributed1)
    model = Model(inputs=seq_input, output=b)
    return model
