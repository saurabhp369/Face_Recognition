#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : neutral vs. facial expression classification
#=======================================================

from Utils.data_utils_2 import *
from Utils.data_utils_1 import *
from Utils.PCA import *
from Utils.MDA import *
from classifiers.bayes import *
from classifiers.kNN import *
from classifiers.kernel_SVM import *
from classifiers.boosted_SVM import *

def main():
    # load the data from mat files
    faces, illum, poses = load_data()
    indices = new_data(faces)
    labels_1 = make_labels()
    labels_2 = make_labels_SVM()
    n_classes = len(set(labels_1))
    x_train = faces[:,:,indices[:300]].reshape((24*21,-1)).T
    x_test = faces[:,:,indices[300:]].reshape((24*21,-1)).T
    y_train = labels_1[:300]
    y_test = labels_1[300:]

    process_choice = int(input('Enter the choice of data compression method 1:PCA 2:MDA: '))

    if process_choice == 1:
         # perform PCA on train and test data
        X_train = PCA(x_train)
        X_test = PCA(x_test)
        print('--------- PCA Done ----------')
    else:
        # perform MDA on the train and test data
        X_train = MDA(x_train,y_train,n_classes)
        X_test = MDA(x_test,y_test,n_classes, 'test')
        print('--------- MDA Done ----------')
    
    # visualise the data
    # visualise_data(x_train[0], X_train[0])

    choice = int(input('Enter the choice of classifier 1:Bayes 2:kNN 3:Boosted SVM 4:kernel_SVM: '))

    if choice == 1:
        ###################### Gaussian Bayes Classifier #####################
        pred  = bayes_classifier(X_train, y_train, X_test, n_classes)
        print('The test accuracy of the bayes classifier is ',np.mean(pred == y_test)*100)

    if choice == 2:
        ########################### kNN Classifier ############################
        neighbors = [1,2,3,4]
        best_model = []
        best_accuracy = -100
        for k in neighbors:
            dist_matrix = calc_euclidean_dist(X_train, X_test)
            pred = predict_labels(dist_matrix, y_train,k)
            acc = np.mean(pred == y_test)*100
            if acc > best_accuracy:
                best_model = [k]
                best_accuracy = acc
        print('The best value of k is ', best_model[0])
        print('The test accuracy of the kNN classifier is ', best_accuracy)        

    if choice == 3:
        ###################### Boosted SVM Classifier #########################
        if process_choice == 2:
            print('Boosted SVM is not able to find an optimal solution with MDA')
        else:
            y_train = labels_2[:300]
            y_test = labels_2[300:]
            iterations = [5, 10, 15]
            best_model = []
            best_accuracy = -100
            for K in iterations:
                pred = boosted_SVM(X_train, X_test, y_train,K )
                acc = np.mean(pred == y_test.reshape(100,1))*100
                if acc > best_accuracy:
                    best_model = [K]
                    best_accuracy = acc
            print('The best value of number of iterations is ', best_model[0])
            print('The test accuracy of the boosted SVM classifier ',best_accuracy)

    if choice == 4:
        ######################## kernel SVM Classifier ########################
        y_train = labels_2[:300]
        y_test = labels_2[300:]
        kernel_choice = int(input('Enter the kernel choice 1:RBF 2:Polynomial '))
        if kernel_choice == 1:
            values = [1, 5, 10, 20]
        else:
            values = [1, 2, 3]
        best_model = []
        best_accuracy = -100
        for p in values:
            pred = kernel_SVM(X_train,y_train,X_test, p, kernel_choice)
            acc = np.mean(pred == y_test.reshape(100,1))*100
            if acc > best_accuracy:
                best_model = [p]
                best_accuracy = acc
        print('The best value of kernel parameter is ', best_model[0])
        print('The test accuracy of the kernel SVM classifier ',best_accuracy)


if __name__ == '__main__':
    main()

