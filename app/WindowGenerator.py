import numpy as np
import tensorflow as tf


class WindowGenerator():
    """
    A class to represent a window object.

    :attr:
        input_width: int
            history size
        label_width: int
            history size
        shift: int
            prediction of 'shift' days into the future

    methods:
        __repr__():
            Returns a string representing window object

    """

    def __init__(self, input_width, label_width, shift,
                 data, label_columns=None):
        """
        Constructs all the necessary attributes for the Window object.
        """

        # Store the raw data
        self.data = data

        # Work out the label column indices
        self.label_columns = label_columns

        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i,
                               name in enumerate(self.data.columns)}

        # Work out the window params
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

        self.data_set = self.make_dataset(self.data)

    def __repr__(self):
        """
        Returns a string representing window object
        """

        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def make_dataset(self, data):
        """
        Takes a time series DataFrame and converts it to a tf.data.Dataset of (input_window, label_window) pairs 
        using the preprocessing.timeseries_dataset_from_array function

        :params:
            data: DataFrame
                time series data

        :return:
            ds: tf.data.Dataset
                time series data 

        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_window)

        return ds

    def split_window(self, features):
        """
        Given a list of consecutive inputs, convert them to a window of inputs and a window of labels.

        :params:
            features: tf tensor
                window object

        :return:
            inputs: tf tensor
                tensor of inputs
            labels: tf tensor
                tensor of labels

        """

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack([
                labels[:, :, self.column_indices[name]] for name in self.label_columns
            ], axis=-1)

        # preserve the shape of the input
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
