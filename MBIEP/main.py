import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import f1_score,confusion_matrix, average_precision_score
from Data import getData
from Model import SPEP


def test_fun(dataset, label, model):
    pred = model(dataset, training=False)
    acc = metrics.BinaryAccuracy()(label, pred)
    pre = metrics.Precision()(label, pred)
    rec = metrics.Recall()(label, pred)
    auc = metrics.AUC()(label, pred)
    ap = average_precision_score(label, pred)

    ypred = tf.math.greater(pred, tf.constant(0.5))
    ypred = tf.keras.backend.eval(ypred)
    tn, fp, fn, tp = confusion_matrix(label, ypred).ravel()
    F1 = f1_score(label, ypred)
    Spe = tn / (tn + fp)
    NPV = tn / (tn + fn)

    print('================begin to test:================')
    print('- Accuracy %.4f' % acc)
    print('- Precision %.4f' % pre)
    print('- Recall %.4f' % rec)
    print('- F1-score %.4f' % F1)
    print('- Specificity %.4f' % Spe)
    print('- NPV %.4f' % NPV)
    print('- AUC %.4f' % auc)
    print('- AP %.4f' % ap)


def plot_loss(loss_values,val_loss_values):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, color = '#FF4747', label='Training loss', linewidth='0.6')
    plt.plot(epochs, val_loss_values, color = '#00BCB4', label='Validation loss', linewidth='0.6')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_acc(acc,val_acc):
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, color = '#FF4747', label='Training acc', linewidth='0.6')
    plt.plot(epochs, val_acc,color = '#00BCB4', label='Validation acc', linewidth='0.6')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # initialize data class, and model class
    data = getData()
    model = SPEP.build(input_shape_gse=[8, 3, 2], input_shape_emb=[64, ], input_shape_sub=[1024, ])
    # specifies the model for calling the CB class
    # set the training parameters
    epochs = 20
    batch_size = 64
    vali_size = 300
    # record the loss and b-acc during training
    tloss = []
    tacc = []
    vloss = []
    vacc = []
    # define loss function and optimizer function
    loss_fun = losses.BinaryCrossentropy(from_logits=False)
    opt_fun = optimizers.RMSprop(learning_rate=0.001)
    # def the acc and loss for training and validation
    train_loss = metrics.Mean()
    train_acc = metrics.BinaryAccuracy()
    vali_loss = metrics.Mean()
    vali_acc = metrics.BinaryAccuracy()

    @tf.function
    def train_fun(dataset, label):
        with tf.GradientTape() as tape:
            pred = model(dataset, training=True)
            loss = loss_fun(label, pred)
        gradient = tape.gradient(loss, model.trainable_variables)
        opt_fun.apply_gradients(zip(gradient, model.trainable_variables))

        train_loss(loss)
        train_acc(label, pred)

    @tf.function
    def vali_fun(dataset, label):
        pred = model(dataset, training=False)
        loss = loss_fun(label, pred)
        vali_loss(loss)
        vali_acc(label, pred)

    print("================begin to train:================")
    for i in trange(epochs):
        # shuffle the data
        data.shuffle()
        # get every batch for training and validation process
        for iter, idx in enumerate(range(0, data.training_size, batch_size)):
            # zip the data together as input
            batch_G = data.train_G[idx:idx + batch_size]
            batch_S = data.train_S[idx:idx + batch_size]
            batch_E = data.train_E[idx:idx + batch_size]
            batch_dict = {'Embedding': batch_E, 'Subloc': batch_S, 'GeneExpression': batch_G}
            batch_Y = data.train_Y[idx:idx + batch_size]

            data.vali_shuffle()
            vali_G = data.vali_G[:vali_size]
            vali_S = data.vali_S[:vali_size]
            vali_E = data.vali_E[:vali_size]
            vali_dict = {'Embedding': vali_E, 'Subloc': vali_S, 'GeneExpression': vali_G}
            vali_Y = data.vali_Y[:vali_size]

            # Reset the states of loss and acc function every epoch
            train_loss.reset_states()
            train_acc.reset_states()
            vali_loss.reset_states()
            vali_acc.reset_states()

            # train and validate the data, and compute the loss、acc, optimize the weights in the model network
            train_fun(batch_dict, batch_Y)
            vali_fun(vali_dict, vali_Y)

            # print the acc、loss every 10 batch
            if iter % 10 == 0:
                # record the loss and acc value to draw the curve
                tloss.append(train_loss.result())
                tacc.append(train_acc.result())
                vloss.append(vali_loss.result())
                vacc.append(vali_acc.result())
                print("=====epoch:%d iter:%d=====" % (i + 1, iter + 1))
                print('- loss: %.4f' % train_loss.result())
                print('- binary_accuracy %.4f' % train_acc.result())
                print('- val_loss %.4f' % vali_loss.result())
                print('- val_binary_accuracy %.4f' % vali_acc.result())
        # transmit log data
        val_logs = {'val_loss': vali_loss.result(), 'val_acc': vali_acc.result()}

    # display loss and acc during training
    plt.style.use('seaborn-ticks')
    plt.rc('font', size=7)
    plt.rcParams['figure.figsize'] = (7, 4)
    plot_loss(tloss, vloss)
    plot_acc(tacc, vacc)

    test_dict = {'Embedding': data.test_E, 'Subloc': data.test_S,
                 'GeneExpression': data.test_G}
    test_fun(test_dict, data.test_Y, model)