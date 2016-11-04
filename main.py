import numpy as np
import pandas as pd
import time
# local library
import utils
from config import random_search

def main():
    print ("Started!!")
    st = time.time()
    symbols = utils.get_sap_symbols('sap500')
    np.random.shuffle(symbols)
    chosen_symbols = symbols[:5]
    start_date="2014-10-01"
    end_date="2016-10-01"
    # use Open data
    input_data = utils.get_data_list_key(chosen_symbols, start_date, end_date)
    target_data = utils.get_data('^GSPC', start_date, end_date)['Open']
    elapsed = time.time() - st
    print ("time for getting data:", elapsed)

    train_st = pd.Timestamp("2014-10-01")
    train_end = pd.Timestamp("2016-04-01")
    test_st = pd.Timestamp("2016-04-02")
    test_end = pd.Timestamp("2016-10-01")

    train_input = input_data.loc[(input_data.index >= train_st) & (input_data.index <= train_end)]
    train_target = target_data.loc[(target_data.index >= train_st) & (target_data.index <= train_end)]
    test_input = input_data.loc[(input_data.index >= test_st) & (input_data.index <= test_end)]
    test_target = target_data.loc[(target_data.index >= test_st) & (target_data.index <= test_end)]

    best_conf = random_search(train_input, train_target, test_input, test_target)
    print("best config:")
    print(best_conf)
    
if __name__ == '__main__':
    main()