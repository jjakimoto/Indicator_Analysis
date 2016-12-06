from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, FactorAnalysis
from utils import testscore

def pca_predict(n, train_input, train_target, test_input, model="pca"):
    if model is "pca":
        pca = PCA()
    elif model is "fa":
        pca = FactorAnalysis()
    regr = LinearRegression()
    Y_test = test_target.values.reshape([-1, 1])
    Y_train = train_target.values.reshape([-1, 1])
    pca.n_components=n
    X = pca.fit_transform(train_input.values)
    regr.fit(X, Y_train)
    X = pca.transform(test_input.values)
    prediction = regr.predict(X)
    return prediction

def get_param(train_input, target_input, valid_input, valid_target, model="pca", n_search=100):
    scores = []
    dim = train_input.values.shape[1]
    step = dim // n_search
    if step == 0:
        step = 1
    n_components = np.arange(1, dim, step)
    Y = valid_target.values.reshape([-1, 1])
    for n in n_components:
        prediction = pca_predict(n, train_input, train_target, valid_input, model)
        sc = testscore(prediction, Y)
        scores.append(sc)
    return n_components[np.argmin(scores)]