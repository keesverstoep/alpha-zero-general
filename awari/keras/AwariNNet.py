import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
# from keras.callbacks import TensorBoard

"""
NeuralNet for the game of Awari.
Currently based on NeuralNet for RTS since that also has feature places
"""

class Configuration:
    class _NNetArgs:
        def __init__(self, num_channels, dropout, lr):
            self.dropout = dropout
            self.lr = lr
            self.num_channels = num_channels  # used by nnet conv layers

    def __init__(self):
        self.nnet_args = self._NNetArgs(num_channels = 64, dropout = 0.3, lr = 0.01)
        # NOTE: lr changed to 0.001 since we seem to be levelling off?
        #self.nnet_args = self._NNetArgs(num_channels = 64, dropout = 0.3, lr = 0.001)
        # self.nnet_args = self._NNetArgs(num_channels = 64, dropout = 0.3, lr = 0.0003)

CONFIG = Configuration()

# def new_run_log_dir(base_dir):
#     log_dir = os.path.join('./log', base_dir)
#    if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     # run_id = len([name for name in os.listdir(log_dir)])
#     # run_log_dir = os.path.join(log_dir, str(run_id))
#     # return run_log_dir
#     return log_dir

class AwariNNet():
    def __init__(self, game, args):
        # game params
        # self.board_x, self.board_y = game.getBoardSize()
        # (self.board_x,) = game.getBoardSize()
        (self.board_x, self.board_y, self.image_stack_size) = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # experiment: do a three-plane encode:
        # plane 0: regular game board
        # plane 1: only flag the other pits that have 1 or 2 stones
        # plane 2: only flag the own pits that have 1 or 2 stones

        # Neural Net
        #self.buildOrig(game)
        self.buildResNet(game)

    def buildOrig(self, game):
        """
        NNet model, copied from Othello NNet, with reduced fully connected layers fc1 and fc2 and reduced nnet_args.num_channels
        :param game: game configuration
        """
        # tensorboard logging:
        # self.log_dir = new_run_log_dir('tensorboard')

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y, self.image_stack_size))  # s: batch_size x board_x x board_y x num_encoders

        x_image = Reshape((self.board_x, self.board_y, self.image_stack_size))(self.input_boards)  # batch_size  x board_x x board_y x image_stack_size
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='same', use_bias=False)(x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))  # batch_size  x board_x x board_y x num_channels
        #h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        #h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='same', use_bias=False)(h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='same', use_bias=False)(h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(CONFIG.nnet_args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(CONFIG.nnet_args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(128, use_bias=False)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(CONFIG.nnet_args.lr))

    def buildResNet(self, game):
        """
        Builds the full Keras model and stores it in self.model.
        """
        mc = self.args

        # (batch, channels, height, width)
        self.input_boards = Input(shape=(self.board_x, self.board_y, self.image_stack_size))  # s: batch_size x board_x x board_y x num_encoders
        x_image = Reshape((self.board_x, self.board_y, self.image_stack_size))(self.input_boards)  # batch_size  x board_x x board_y x image_stack_size

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x_image)
        x = BatchNormalization(axis=3, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.residual_block_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x
        
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=3, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(self.action_size, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="pi")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=3, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="v")(x)

        # self.model = Model(in_x, [policy_out, value_out], name="chess_model")
        # self.model = Model(in_x, [policy_out, value_out], name="awari_model")
        self.model = Model(inputs = self.input_boards, outputs = [policy_out, value_out])
        self.model.summary()
        
        """
        Compiles the model to use optimizer and loss function tuned for supervised learning
        """
        opt = Adam(mc.lr)
        # opt = SGD(lr=mc.lr, momentum=0.9)
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised 
        self.model.compile(optimizer=opt, loss=losses) #, loss_weights=mc.trainer_loss_weights)


    def _build_residual_block(self, x, index):
        mc = self.args
        in_x = x
        res_name = "res"+str(index)

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=3, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=3, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

