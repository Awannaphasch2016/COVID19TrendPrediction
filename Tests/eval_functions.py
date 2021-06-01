import pandas as pd 

def test_mape():
    from Utils.eval_funcs import mape 
    test_data = pd.read_csv('Data/Test/eval_funcs_test_data.csv', index_col=False).to_numpy()
    r = mape(test_data[:,0], test_data[:,1])
    print(r) 

def test_mse():
    from Utils.eval_funcs import mse 
    test_data = pd.read_csv('Data/Test/eval_funcs_test_data.csv', index_col=False).to_numpy()
    r = mse(test_data[:,0], test_data[:,1])
    print(r) 

def test_rmse():
    from Utils.eval_funcs import rmse 
    test_data = pd.read_csv('Data/Test/eval_funcs_test_data.csv', index_col=False).to_numpy()
    r = rmse(test_data[:,0], test_data[:,1])
    print(r) 

def test_r2score():
    from Utils.eval_funcs import r2score 
    test_data = pd.read_csv('Data/Test/eval_funcs_test_data.csv', index_col=False).to_numpy()
    r = r2score(test_data[:,0], test_data[:,1])
    print(r) 

if __name__ == '__main__':
    test_mape()
    test_mse()
    test_rmse()
    test_r2score()
