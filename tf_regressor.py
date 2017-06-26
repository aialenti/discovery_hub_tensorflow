import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from sklearn import preprocessing



important_features = ['X384', 'X3_trans', 'X4_trans', 'X5_trans', 'X134', 'X1_trans', 'X385',
       'X261', 'X0_trans', 'X27', 'X277', 'X45', 'X74', 'X230', 'X52', 'X29',
       'X68', 'X236', 'X49', 'X35', 'X343', 'X44', 'X340', 'X135', 'X199',
       'X337', 'X62', 'X380', 'X218', 'X24', 'X186', 'X140', 'X56', 'X216',
       'X349', 'X352', 'X322', 'X161', 'X156', 'X10']
DATA_FOLDER = 'data/'
#   0.51
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
batch_size = 5000
hm_epochs = 1000

features_nr = len(important_features)

print('Loading dataset...')
train_df = pd.read_csv(os.path.join(DATA_FOLDER,'train_transformed.csv'))
test_df = pd.read_csv(os.path.join(DATA_FOLDER,'test_transformed.csv'))

train_df_f = train_df[important_features+['y_trans']]
test_df_f = test_df[important_features]
print('Done.')


#   Train
train_X = train_df_f.drop('y_trans', axis=1).as_matrix()
train_y = train_df_f['y_trans'].as_matrix()
train_y = train_y.reshape((len(train_y), 1))

#   Test
test_X = test_df_f[important_features].as_matrix()


x = tf.placeholder('float', [None, len(important_features)])
y = tf.placeholder('float')

def neural_network_model(data):
    h1_definition = {'weights':tf.Variable(tf.random_normal([features_nr, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    h2_definition = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    h3_definition = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, 1])),
                    'biases':tf.Variable(tf.random_normal([1]))}

    l1 = tf.add(tf.matmul(data, h1_definition['weights']), h1_definition['biases'])
    l1 = tf.nn.tanh(l1)
    l1 = tf.nn.dropout(l1,0.95)

    l2 = tf.add(tf.matmul(l1, h2_definition['weights']), h2_definition['biases'])
    l2 = tf.nn.tanh(l2)
    l2 = tf.nn.dropout(l2, 0.95)

    l3 = tf.add(tf.matmul(l2, h3_definition['weights']), h3_definition['biases'])
    l3 = tf.nn.tanh(l3)
    l3 = tf.nn.dropout(l3, 0.95)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction)))))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    train_X_chunks = np.array_split(train_X, int(np.ceil(len(train_X) / batch_size)))
    train_y_chunks = np.array_split(train_y, int(np.ceil(len(train_X) / batch_size)))

    epochs = []
    losses = []
    last_session = 0;
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for c in range(int(len(train_X_chunks))):
                epoch_x = train_X_chunks[c]
                epoch_y = train_y_chunks[c]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            epochs.append(epoch)
            losses.append(epoch_loss)
        last_session = sess
        saver.save(sess, 'models/model.ckpt')
    print('Plotting loss...')
    plt.plot(epochs, losses, 'ro-')
    for a,b in zip(epochs, losses):
        plt.text(a, b, str(a))
    plt.show()
    return prediction

def predict(data, model):
    # build your model (same as training)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'models/model.ckpt')
    fd = {x: data}
    pred = model.eval(fd,sess)
    print(pred)
    pred = pd.DataFrame(pred)
    pred[pred[0].isnull()] = np.mean(pred[0])
    mms = preprocessing.MinMaxScaler()
    mms = mms.fit(train_df['y'])
    #pred = mms.inverse_transform(pred)
    print(pred)
    return pred


model = train_neural_network(x)
predictions = predict(test_X, model)

test_df['y'] = predictions
submission = test_df[['ID','y']]
print(submission)
submission.to_csv('submissions/sub.csv',index=False)
