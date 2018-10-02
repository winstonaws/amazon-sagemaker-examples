import tensorflow as tf
import os
import json
import argparse


if __name__ =='__main__':
    # Training environment metadata is available from environment variables.
    # Lots of additional information available not used in this example, such as # gpus/cpus.
    # Easy to interact with once loaded from json, since it's just a regular python dict.
    train_env = json.loads(os.environ['SM_TRAINING_ENV'])
    print(train_env)

    # You can read script parameters either from SM_TRAINING_ENV, or by using argparse.
    # Both approaches demonstrated here.
    foo = float(train_env['hyperparameters']['foo'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--bar', type=float)
    args, unknown = parser.parse_known_args()
    bar = args.bar

    # Load training data file. SageMaker downloads it from S3 for you, and makes it
    # accessible from the local file system.
    train_data_dir = train_env['channel_input_dirs']['training']
    data_file = os.path.join(train_data_dir, 'data.txt')
    with open(data_file, 'r') as f:
        print(f.readlines())
    
    # Use script parameters to do a trivial TensorFlow operation using eager execution,
    # similar to example from: https://www.tensorflow.org/guide/eager#setup_and_basic_usage
    tf.enable_eager_execution()

    m = tf.matmul([[foo]], [[bar]])
    print("hello, {}".format(m))  # => "hello, [[12]]"