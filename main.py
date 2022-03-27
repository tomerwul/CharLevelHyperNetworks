import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from utils import CharLevel_Dataset, stratified_split
import argparse
import numpy as np
np.random.seed(42)


def classification_rep(y_true, preds):
    pred_labels = [1 if x > 0.5 else 0 for x in preds]
    print(classification_report(y_true, pred_labels, digits=3))


class DynamicHyperNetwork(layers.Layer):
    """
    A dynamic HyperNetwork layer, generating weights for convolutional layers.
    """
    def __init__(self,
                 input_dim=32,
                 z_dim=4,
                 name='dynamic_hypernetwork',
                 kernel_size=None,
                 num_filters=1,
                 enc_units=32,
                 batch_size=32,
                 num_layers=3,
                 max_pool=3,
                 encoded=10):
        super(DynamicHyperNetwork, self).__init__(name=name)
        self.gru = layers.Bidirectional(layers.GRU(encoded, return_sequences=False, return_state=True))
        self.encoded = encoded
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.softmax = layers.Activation('softmax')
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.enc_units = enc_units
        self.globalmax = layers.GlobalMaxPool1D()
        self.activation = layers.Activation('relu')
        z_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        b_init = tf.zeros_initializer()

        self.z = []

        for j in range(num_layers):
            self.z.append(tf.Variable(name="hyper_z",
                                      initial_value=z_init(shape=(1, z_dim), dtype='float32'),
                                      trainable=True))
        self.hyper_a = layers.Dense(self.encoded * num_filters, use_bias=False)
        self.hyper_b = layers.Dense(kernel_size * num_filters, use_bias=False)
        self.conv_biases = tf.Variable(name="conv_biases",
                                       initial_value=b_init(shape=(num_filters),
                                                            dtype='float32'), trainable=True)
        self.maxpool = layers.MaxPooling1D(pool_size=max_pool)

    def call(self, inputs, layer=None):
        if layer == 0:
            pad_side = self.num_filters - inputs.get_shape()[2]
            padding = [[0, 0], [0, 0], [int(pad_side / 2), int(pad_side / 2)]]
            inputs = tf.pad(inputs, padding)

        results = []
        gru = self.gru(inputs)
        c_vectors = tf.split(gru[0], self.batch_size, 0, name='split1')
        input_vectors = tf.split(inputs, self.batch_size, 0, name='split2')
        for i, c in enumerate(c_vectors):

            filters_a = self.hyper_a(c)
            filters_a = tf.reshape(filters_a, [self.num_filters, self.encoded], name='reshape_1')
            filters_b = self.hyper_b(filters_a)
            filters_b = tf.reshape(filters_b, (self.num_filters, self.num_filters, self.kernel_size))
            filters = tf.transpose(filters_b)

            Y = tf.nn.conv1d(input_vectors[i], filters, padding='SAME', stride=1)
            Y = Y + self.conv_biases
            results.append(Y)

        out = self.activation(tf.concat(results, 0))
        out = self.maxpool(out)
        return out


class StaticHyperNetwork(layers.Layer):
    """
    A static HyperNetwork layer, generating weights for convolutional layers.
    """
    def __init__(self,
                 input_dim=32,
                 z_dim=4,
                 name='static_hypernetwork',
                 kernel_size=None,
                 num_filters=1,
                 enc_units=32,
                 batch_size=32,
                 num_layers=3,
                 max_pool=3):
        super(StaticHyperNetwork, self).__init__(name=name)
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.softmax = layers.Activation('softmax')
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.enc_units = enc_units
        self.globalmax = layers.GlobalMaxPool1D()
        self.activation = layers.Activation('relu')
        z_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        b_init = tf.zeros_initializer()
        self.z = []

        for j in range(num_layers):
            self.z.append(tf.Variable(name="hyper_z",
                                      initial_value=z_init(shape=(1, self.z_dim), dtype='float32'),
                                      trainable=True))

        self.hyper_a = layers.Dense(self.z_dim * self.num_filters, use_bias=False)
        self.hyper_b = layers.Dense(self.kernel_size * self.num_filters,  use_bias=False)

        self.conv_biases = tf.Variable(name="conv_biases",
                                       initial_value=b_init(shape=(self.num_filters),
                                        dtype='float32'), trainable=True)
        self.maxpool = layers.MaxPooling1D(pool_size=max_pool)

    def call(self, inputs, layer=None):

        if layer == 0:
            pad_side = self.num_filters - inputs.get_shape()[2]
            padding = [[0, 0], [0, 0], [int(pad_side / 2), int(pad_side / 2)]]
            inputs = tf.pad(inputs, padding)

        filters_a = self.hyper_a(self.z[layer])
        filters_a = tf.reshape(filters_a, [self.num_filters, self.z_dim], name='reshape_1')
        filters_b = self.hyper_b(filters_a)
        filters_b = tf.reshape(filters_b, (self.num_filters, self.num_filters, self.kernel_size))
        filters = tf.transpose(filters_b)

        Y = tf.nn.conv1d(inputs, filters, padding='SAME', stride=1)
        Y = Y + self.conv_biases
        out = self.activation(Y)
        out = self.maxpool(out)
        return out


class HyperClassifier(tf.keras.Model):
    def __init__(self,
                 vocab_size=69,  # 68 characters + 1
                 input_length=100,
                 embedding_dim=30,
                 num_filters=64,
                 name='HyperCNNclassfier',
                 cnn_layers=3,
                 ker_size=4,
                 fc_layers=2,
                 max_pool=3,
                 z_dim=4,
                 d=[128, 32],
                 p_dropout=0.2,
                 model_type='static'):
        super(HyperClassifier, self).__init__(name=name)

        self.num_cnn_layers = cnn_layers
        self.num_fc_layers = fc_layers
        self.input_layer = tf.keras.layers.Input(shape=(None,), name='Input_Word', )
        self.embedding_layer = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim,
                                                input_length=input_length)
        self.activation = layers.ReLU()
        self.fc_layers = []
        if model_type == 'static':
            self.conv1d_hypernetwork = StaticHyperNetwork(input_dim=embedding_dim,
                                                          kernel_size=ker_size,
                                                          max_pool=max_pool,
                                                          num_filters=num_filters,
                                                          z_dim=z_dim,
                                                          num_layers=cnn_layers)
        else:
            self.conv1d_hypernetwork = DynamicHyperNetwork(input_dim=embedding_dim,
                                                          kernel_size=ker_size,
                                                          max_pool=max_pool,
                                                          num_filters=num_filters,
                                                          z_dim=z_dim,
                                                          num_layers=cnn_layers)
        for i in range(self.num_fc_layers):
            self.fc_layers.append(layers.Dense(d[i], activation='relu', name='dense_{}'.format(i)))

        self.dropout = layers.Dropout(p_dropout)
        self.dense3 = layers.Dense(1, activation='sigmoid', name='dense_out')
        self.flatten = layers.Flatten()

    def call(self, inputs):
        conv = self.embedding_layer(inputs)

        for j in range(0, self.num_cnn_layers):
            conv = self.conv1d_hypernetwork(conv, layer=j)

        dense = self.flatten(conv)
        for d in range(0, self.num_fc_layers):
            dense = self.fc_layers[d](dense)
            dense = self.activation(dense)
            dense = self.dropout(dense)

        output = self.dense3(dense)
        return output


def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Parse Arguments.')
    parser.add_argument('--train_path', help='Path to training data.')
    parser.add_argument('--test_path', help='Path to test data.')
    parser.add_argument('--num_train', help='Number of training samples.')
    parser.add_argument('--num_valid', help='Number of validation samples.')
    parser.add_argument('--model', help='Hypernetwork type (static, dynamic).')
    parser.add_argument('--epochs', default=10, help='Number of training epochs.')
    parser.add_argument('--lc', default=True, help='Lower-case input data.')
    parser.add_argument('--batch_size', default=32, help='Batch size.')
    parser.add_argument('--emb_dim', default=50, help='Embedding layer dimension.')
    parser.add_argument('--ker_size', default=7, help='Convolutional layers kernel size.')
    parser.add_argument('--pool_size', default=4, help='Pool size of max-pooling layer.')
    parser.add_argument('--dropout_p', default=0.5, help='Dropout probability.')
    parser.add_argument('--min_freq', default=1, help='Minimum number of token apearences.')
    parser.add_argument('--num_filters', default=64, help='Number of filters in conv. layers.')
    parser.add_argument('--cnn_layers', default=2, help='Number of CNN layers of the main model.')
    parser.add_argument('--z_dim', default=10, help='The dimension of the hypernetwork layer embedding vector..')
    parser.add_argument('--dense_layers', default=2, help='Number of fully-connected layers of the main model.')
    parser.add_argument('--max_length', default=120, help='Maximum characters in input sequences.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    X_train, y_train = CharLevel_Dataset(file=args.train_path,
                                         maxlen=args.max_length,
                                         lowercase=args.lc, min_freq=args.min_freq)
    X_test, y_test = CharLevel_Dataset(file=args.test_path,
                                       maxlen=args.max_length,
                                       lowercase=args.lc, min_freq=args.min_freq)

    tf.keras.backend.clear_session()
    X_train, y_train, X_val, y_val = stratified_split(X_train, y_train,  train_size=args.num_train, test_size=args.num_valid)

    metrics = [
        tf.keras.metrics.TruePositives(name='true_positives', thresholds=0.5),
        tf.keras.metrics.FalseNegatives(name='false_negatives', thresholds=0.5),
        tf.keras.metrics.FalsePositives(name='false_positives', thresholds=0.5),
        tf.keras.metrics.TrueNegatives(name='true_negatives', thresholds=0.5),
        tf.keras.metrics.BinaryAccuracy(name='accuracy0.5', threshold=0.5),
        tf.keras.metrics.Precision(name='precision0.5', thresholds=0.5),
        tf.keras.metrics.Recall(name='recall0.5', thresholds=0.5),
    ]

    model = HyperClassifier(vocab_size=69,
                            input_length=args.max_length,
                            embedding_dim=args.emb_dim,
                            num_filters=args.num_filters,
                            name='hateclassfier',
                            max_pool=args.pool_size,
                            cnn_layers=args.cnn_layers,
                            fc_layers=args.dense_layers,
                            ker_size=args.ker_size,
                            z_dim=args.z_dim,
                            p_dropout=args.dropout_p,
                            model_type=args.model)

    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=[metrics])
    model.fit(X_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs, verbose=2)

    preds = model.predict(X_test, verbose=1)
    classification_rep(y_true=y_test, preds=preds)


if __name__ == "__main__":
    main()


