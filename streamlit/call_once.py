import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sagemaker.s3 import S3Downloader
from sagemaker.tensorflow import TensorFlowPredictor

local_folder_path = '/data/'
endpoint_name = ":+:+:+:+:+:+:+:+:+:+"
bucket_name = ":+:+:+:+:+:+:+:+:+:+"
n_user = 610
n_item = 9724
batch_size = 100
threshold = 0.5


def artifact_download():
    original_data = f's3://{bucket_name}{local_folder_path}'
    org_list = S3Downloader.list(original_data)
    os.makedirs(f'.{local_folder_path}', exist_ok=True)
    for full_path in org_list:
        print(full_path)
        S3Downloader.download(full_path, local_path=f'.{local_folder_path}')


def load_testing_data(base_dir):
    """ load testing data """
    df_test = np.load(os.path.join(base_dir, 'test.npy'))
    user_test, item_test, y_test = np.split(np.transpose(df_test).flatten(), 3)
    return user_test, item_test, y_test


if __name__ == '__main__':
    artifact_download()
    # read testing data
    user_test, item_test, test_labels = load_testing_data(f'.{local_folder_path}')

    # one-hot encode the testing data for model input
    with tf.compat.v1.Session() as tf_sess:
        test_user_data = tf_sess.run(tf.one_hot(user_test, depth=n_user)).tolist()
        test_item_data = tf_sess.run(tf.one_hot(item_test, depth=n_item)).tolist()

    predictor = TensorFlowPredictor(endpoint_name)

    # make batch prediction
    y_pred = []
    for idx in range(0, len(test_user_data), batch_size):
        # reformat test samples into tensorflow serving acceptable format
        input_vals = {
            "instances": [
                {'input_1': u, 'input_2': i}
                for (u, i) in zip(test_user_data[idx:idx + batch_size], test_item_data[idx:idx + batch_size])
            ]}

        # invoke model endpoint to make inference
        pred = predictor.predict(input_vals)

        # store predictions
        y_pred.extend([i[0] for i in pred['predictions']])

    # let's see some prediction examples, assuming the threshold
    # --- prediction probability view ---
    print('This is what the prediction output looks like')
    print(y_pred[:5], end='\n\n\n')

    # --- user item pair prediction view, with threshold of 0.5 applied ---
    pred_df = pd.DataFrame([
        user_test,
        item_test,
        (np.array(y_pred) >= threshold).astype(int)],
    ).T

    pred_df.columns = ['userId', 'movieId', 'prediction']

    print('We can convert the output to user-item pair as shown below')
    print(pred_df.head(), end='\n\n\n')

    # --- aggregated prediction view, by user ---
    print('Lastly, we can roll up the prediction list by user and view it that way')
    print(pred_df.query('prediction == 1').groupby('userId').movieId.apply(list).head().to_frame(), end='\n\n\n')