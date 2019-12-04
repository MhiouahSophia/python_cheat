import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
from numba import cuda
from keras import backend as K
from keras import backend as bek
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard



############## CROSS VALIDATION KERAS 
def train_model_cv(model_initial, n_split, datasets_dir, output_dir_path, tensorboard_log_path=None,
                   dataset_fold_id=None,
                   epochs=50, period=1, batch_size=128, learning_rate=1e-3, min_learning_rate=1e-4, patience=10,
                   use_best_weights=True, output_weights_name='best_weights',
                   monitor='val_loss', optimization_mode='auto'):
    X_train_inital, y_train_initial, X_test, y_test_initial = load_dataset_at(datasets_dir, cross_validation=True,
                                                                              fold_index=dataset_fold_id)

    # get list if unique track ID
    track_unique_id_train = X_train_inital.shape[0]
    i = 0
    dict_score = {}

    for train_idx, valid_idx in KFold(n_split).split(range(track_unique_id_train)):

        i = i + 1
        model = model_initial
        print('SPLIT NUMBER: ' + str(i))
        # map the list of train/test
        X_train = X_train_inital[train_idx]
        X_valid = X_train_inital[valid_idx]

        y_train = y_train_initial[train_idx]
        y_valid = y_train_initial[valid_idx]

        print(X_train.shape)
        print(X_valid.shape)
        print(X_test.shape)

        class_weight = get_class_weight(y_train)

        y_train = to_categorical(y_train, len(np.unique(y_train)))
        y_valid = to_categorical(y_valid, len(np.unique(y_valid)))
        y_test = to_categorical(y_test_initial, len(np.unique(y_test_initial)))

        print(y_train.shape)
        print(y_valid.shape)
        print(y_test.shape)

        factor = 1. / np.cbrt(2)  # factor = 1. / np.sqrt(2)

        check_dir("%s/weights/" % output_dir_path)
        if use_best_weights:
            weight_fn = "%s/weights/%s_on_fold_%s.h5" % (output_dir_path, output_weights_name, str(i))
        else:
            weight_fn = "%s/weights/%s_{epoch:02d}.h5/" % (
                output_dir_path, output_weights_name)

        # Callbacks
        model_checkpoint = ModelCheckpoint(weight_fn,
                                           verbose=1,
                                           mode=optimization_mode,
                                           monitor=monitor,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           period=period)

        reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                      patience=patience,
                                      mode=optimization_mode,
                                      factor=factor,
                                      cooldown=0,
                                      min_lr=min_learning_rate,
                                      verbose=2)

        tensorboard = TensorBoard(log_dir='./log' if not tensorboard_log_path else tensorboard_log_path,
                                  batch_size=batch_size)

        callback_list = [model_checkpoint, reduce_lr, tensorboard]

        optm = Adam(lr=learning_rate)

        model.compile(optimizer=optm, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history_ = model.fit(X_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=callback_list,
                             class_weight=class_weight,
                             verbose=2,
                             validation_data=(X_valid, y_valid))

        check_dir("%s/loglosses/" % output_dir_path)
        save_dataframe_as_csv(pd.DataFrame(history_.history['loss']), out_path="%s/loglosses/" % (output_dir_path),
                              filename='loss_history_{0}'.format(i))

        save_dataframe_as_csv(pd.DataFrame(history_.history['val_loss']), out_path="%s/loglosses/" % (output_dir_path),
                              filename='val_loss_history_{0}'.format(i))

        print("\nEvaluating : ")
        loss_test, accuracy_test = model.evaluate(X_test, y_test, batch_size=batch_size)
        print()
        print("Final Accuracy - Test: ", accuracy_test)
        print("Final Loss - Test: ", loss_test)

        loss_valid, accuracy_valid = model.evaluate(X_valid, y_valid, batch_size=batch_size)
        print()
        print("Final Accuracy - Valid: ", accuracy_valid)
        print("Final Loss - Valid: ", loss_valid)

        loss_train, accuracy_train = model.evaluate(X_train, y_train, batch_size=batch_size)
        print()
        print("Final Accuracy - Train: ", accuracy_train)
        print("Final Loss - Train: ", loss_train)

        check_dir("%s/" % (output_dir_path))
        summary_filename = "%s/ModelSummary.txt" % (output_dir_path)
        with open(summary_filename, 'w') as arch_file:
            with redirect_stdout(arch_file):
                model.summary()

        dict_score['model_on_fold_{0}'.format(i)] = [accuracy_train, accuracy_valid, accuracy_test, loss_train,
                                                     loss_valid,
                                                     loss_test]
        print(dict_score)
        del X_train
        del X_valid
        del history_
        del model

    dict_score['Mean'] = [np.mean([dict_score[i][0] for i in dict_score.keys()]), np.mean([dict_score[i][1] for i in dict_score.keys()]), np.mean([dict_score[i][2] for i in dict_score.keys()]), np.mean([dict_score[i][3] for i in dict_score.keys()]),
                                                 np.mean([dict_score[i][4] for i in dict_score.keys()]),
                                                 np.mean([dict_score[i][5] for i in dict_score.keys()])]

    dict_score['Ecart-type'] = [np.std([dict_score[i][0] for i in dict_score.keys()]), np.std([dict_score[i][1] for i in dict_score.keys()]), np.std([dict_score[i][2] for i in dict_score.keys()]), np.std([dict_score[i][3] for i in dict_score.keys()]),
                                                 np.std([dict_score[i][4] for i in dict_score.keys()]),
                                                 np.std([dict_score[i][5] for i in dict_score.keys()])]

    score_df = pd.DataFrame.from_dict(dict_score, orient='index',
                                      columns=['accuracy_train', 'accuracy_train', 'accuracy_test',
                                               'loss_train',
                                               'loss_valid',
                                               'loss_test'])

    score_df.to_csv(str(output_dir_path) + '/score_for_cross_validation.csv')
    print(score_df)
    return score_df
