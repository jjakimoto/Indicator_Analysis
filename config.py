import numpy as np
import random
import time
# local libarary
from model import MLP
    
def random_search(train_input, train_target, valid_input, valid_target,
        save_path, device='/gpu:0', n_iter=100, data_name='sap500'):
    """Randome Search for Hyper parameter of the model
    
    Args:
        train(valid)_input, train(valid)_target (DataFrame): data to train and validation of network
        n_inter(int): the number to iterate for parameter search
        data_name(str): the name of target data
    """
    n_stock = len(train_input.values[0])
    conf = SearchConfig()
    config_list = []
    loss_list = []
    st = time.time()
    chosen_symbols = train_input.columns
    best_loss = np.inf
    st = time.time()
    for i in range(n_iter):
        try:
            random_conf = generate_config(conf, n_stock, device, save_path)
            mlp = MLP(random_conf)
            print('number:', i)
            mlp.training(train_input, train_target)
            loss = mlp.accuracy(valid_input, valid_target)
            print('loss:', loss)
            if best_loss > loss:
                # mlp.save()
                best_loss = loss
                best_conf = random_conf
                print(best_conf)
                prediction = mlp.predict(valid_input)
                prediction.to_csv("%s_%d.csv" % (data_name, len(chosen_symbols)))
                plt.plot(prediction, label='prediction')
                plt.plot(valid_target, label='target')
                plt.title('%s_%d' % (data_name, len(chosen_symbols)))
                plt.legend()
                plt.savefig('%s_%d.png' % (data_name, len(chosen_symbols)))
                plt.close()
            elapsed = time.time() - st
            print ("elapsed time:", elapsed)
            print('******************************************')
            config_list.append(random_conf)
            loss_list.append(loss)
        except KeyboardInterrupt:
            break
        except:
            pass
    return best_conf
    
    
class SearchConfig(object):
    """Search Range for random search"""
    n_layer = [1, 8]
    n_hidden=[10, 1000]
    drop_ratio=[0, 0.5]
    learning_rate=[1.0e-5, 0.1]
    anneal=[50, 1.0e4]

    
def generate_config(conf, n_stock, device='/cpu:0', save_path='/path/to/your/save/directory/'):
    """Make MLP configuration randomly"""
    n_layer = np.random.random_integers(conf.n_layer[0], conf.n_layer[1])
    random_conf = {}
    model = []
    for i in range(n_layer):
        state = {}
        state['n_hidden'] = int(sample_geo(conf.n_hidden))
        state['is_batch'] = sample_TF()
        state['is_drop'] = sample_TF()
        state['drop_rate'] = np.random.uniform(conf.drop_ratio[0], conf.drop_ratio[1])
        model.append(state)
    random_conf['model'] = model
    random_conf['learning_rate'] = sample_geo(conf.learning_rate)
    random_conf['anneal'] = sample_geo(conf.anneal)
    random_conf['device'] = device
    random_conf['save_path'] = save_path
    random_conf['n_stock'] = n_stock
    random_conf['n_batch'] = 64
    random_conf['n_epochs'] = 500
    random_conf['is_load'] = False
    return random_conf


def sample_geo(conf):
    """Sample logscale"""
    low = conf[0]
    high = conf[1]
    u = np.random.uniform()
    return np.exp((np.log(high) - np.log(low)) * u + np.log(low))


def sample_TF():
    """Sample Truth or False"""
    tf = np.random.randint(2)
    if tf == 0:
        return False
    else:
        return True
