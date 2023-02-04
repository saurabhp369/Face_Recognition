#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : identifying the subject label from a test image
#=======================================================

from Utils.data_utils_1 import *
from Utils.PCA import *
from Utils.MDA import *
from classifiers.bayes import *
from classifiers.kNN import *

def main():
    # load the data from mat files
    faces, illum, poses = load_data()

    # create labels for the faces
    face_labels,_,_ = create_labels()
    n_classes = len(set(face_labels))

    # split the data into train and test set
    train_ind, test_ind = train_test_split()
    x_train  = faces[:,:,train_ind].reshape((24*21,-1)).T
    y_train = face_labels[train_ind,]
    x_test  = faces[:,:,test_ind].reshape((24*21,-1)).T
    y_test = face_labels[test_ind,]

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

    # Enter the choice of the classifier
    choice = int(input('Enter the choice of classifier 1:Bayes 2:kNN: '))

    if choice == 1:
        ###################### Gaussian Bayes Classifier #####################
        pred  = bayes_classifier(X_train, y_train, X_test, n_classes)
        print('The test accuracy of the bayes classifier is ',np.mean(pred == y_test)*100)

    if choice == 2:
        ########################### kNN Classifier ############################
        neighbors = [1,2,3,4] # chosen values for k
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



if __name__ == '__main__':
    main()



