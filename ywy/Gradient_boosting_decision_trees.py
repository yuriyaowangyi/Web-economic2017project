import numpy as np
import sklearn as sk
import sklearn.ensemble.gradient_boosting as GB
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV



def auto_GBR(data,test_ratio = 0.2,features_n = 10, regression = False):

    data = np.asarray(data)
    Y = data[:,0]
    X = data[:,1:]

    print('selecting features...')

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV()
    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(X, Y)
    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > features_n:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    X = X_transform

    print('spliting data...')

    split_line = int(len(Y)*(1-test_ratio))
    x_train, y_train = X[:split_line], Y[:split_line]
    x_test, y_test = X[split_line:], Y[split_line:]

    if regression==True:
        model = GB.GradientBoostingRegressor

        split = [0.5,2,3]
        leaf = [1,2,3]
        rate = [0.01,0.1,0.5]
        dep = [2,3,4]

        Model_error = 10000

        for i in split:
            for j in leaf:
                for k in rate:
                    for z in dep:
                        regressor = model(min_samples_split= i, min_samples_leaf=j, learning_rate=k, max_depth= z)
                        regressor.fit(X=x_train, y= y_train)
                        test_error = sum((regressor.predict(X=x_test)- y_test)**2)
                        if test_error < Model_error:
                            print('=========================')
                            print('Better model with test error:')
                            print(test_error)
                            Model_error = test_error
                            print('param:')
                            param = [i,j,k,z]
                            print(param)
                            print('=========================')
                        else:
                            pass

        print('finish searching...')

        final_regressor = model(min_samples_split= param[0], min_samples_leaf=param[1], learning_rate=param[2], max_depth= param[3])
        final_regressor.fit(X=x_train, y= y_train)
        return final_regressor
    else:
        model = GB.GradientBoostingClassifier

        split = [0.5,2,3]
        leaf = [1,2,3]
        rate = [0.01,0.1,0.5]
        dep = [2,3,4]

        Model_error = 1

        for i in split:
            for j in leaf:
                for k in rate:
                    for z in dep:
                        classifier = model(min_samples_split= i, min_samples_leaf=j, learning_rate=k, max_depth= z)
                        classifier.fit(X=x_train, y= y_train)
                        test_error = sum(classifier.predict(X=x_test) != y_test)/ len(y_test)
                        if test_error < Model_error:
                            print('=========================')
                            print('Better model with test error:')
                            print(test_error)
                            Model_error = test_error
                            print('param:')
                            param = [i,j,k,z]
                            print(param)
                            print('=========================')
                        else:
                            pass

        print('finish searching...')

        final_classifier = model(min_samples_split= param[0], min_samples_leaf=param[1], learning_rate=param[2], max_depth= param[3])
        final_classifier.fit(X=x_train, y= y_train)
        prediction = final_classifier.predict(X= x_test)

        return final_classifier, prediction
