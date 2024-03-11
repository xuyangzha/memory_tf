from keras.models import load_model
import _pickle as pickle
import os
from collections import Counter
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def create_test_set_AWF_disjoint(features_model, n_instance, max_n, n_shot):

    # Load data
    site_dict = {}
    dataset_dir = '../../dataset/extracted_AWF100/'
    sites = os.listdir(dataset_dir)
    random.shuffle(sites)

    for s in sites:
        site_dict[s] = []
        site_random = list(range(1, n_instance+1))
        random.shuffle(site_random)
        site_random = site_random[:n_instance]
        for ins in site_random:
            ins_num = '{:04}'.format(ins)
            file_name = '%s_%s.pkl' % (s, ins_num)
            with open(dataset_dir + '/' + s + '/' + file_name, 'rb') as handle:
                test_set = pickle.load(handle,encoding='iso-8859-1')
                site_dict[s].append(test_set)

    # create_signature and test_set
    signature_dict = {}
    test_dict = {}
    for s in sites:
        data_set = site_dict[s]
        random.shuffle(data_set)
        signature = []
        for i in range(n_shot):
            signature.append(data_set.pop(0))
        signature_dict[s] = signature
        test_dict[s] = data_set[:n_instance - max_n]

    # Feed signature vector to the model to create embedded signature feature's vectors
    signature_vector_dict = {}
    signature_mev_vector_dict = {}
    for i in sites:
        signature_instance = signature_dict[i]
        signature_instance = np.array(signature_instance)
        signature_instance = signature_instance.astype('float32')
        signature_instance = signature_instance[:, :, np.newaxis]
        signature_vector = features_model.predict(signature_instance)
        signature_mev_vector_dict[i]=np.array([signature_vector.mean(axis=0)])
        signature_vector_dict[i] = signature_vector

    # Feed test vector to the model to create embedded test feature's vectors
    test_vector_dict = {}
    for i in sites:
        test_instance = test_dict[i]
        test_instance = np.array(test_instance)
        test_instance = test_instance.astype('float32')
        test_instance = test_instance[:, :, np.newaxis]
        test_vector = features_model.predict(test_instance)
        test_vector_dict[i] = test_vector

    return signature_mev_vector_dict,signature_vector_dict, test_vector_dict


def kNN_accuracy(signature_vector_dict, test_vector_dict, size_of_problem, n_shot):
    X_train = []
    y_train = []

    # print "Size of problem :", size_of_problem
    site_labels = list(signature_vector_dict.keys())
    random.shuffle(site_labels)
    tested_sites = site_labels[:size_of_problem]
    for s in tested_sites:
        for each_test in range(len(signature_vector_dict[s])):
            X_train.append(signature_vector_dict[s][each_test])
            y_train.append(s)

    X_test = []
    y_test = []
    for s in tested_sites:
        for i in range(len(test_vector_dict[s])):
            X_test.append(test_vector_dict[s][i])
            y_test.append(s)

    knn = KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')
    knn.fit(X_train, y_train)

    acc_knn_top1 = accuracy_score(y_test, knn.predict(X_test))
    acc_knn_top1 = float("{0:.15f}".format(round(acc_knn_top1, 6)))
    # Top-2
    count_correct = 0
    for s in range(len(X_test)):
        test_example = X_test[s]
        class_label = y_test[s]
        predict_prob = knn.predict_proba([test_example])
        best_n = np.argsort(predict_prob[0])[-2:]
        class_mapping = knn.classes_
        top_n_list = []
        for p in best_n:
            top_n_list.append(class_mapping[p])
        if class_label in top_n_list:
            count_correct = count_correct + 1

    acc_knn_top2 = float(count_correct) / float(len(X_test))
    acc_knn_top2 = float("{0:.15f}".format(round(acc_knn_top2, 6)))

    # Top 5
    count_correct = 0
    for s in range(len(X_test)):
        test_example = X_test[s]
        class_label = y_test[s]
        predict_prob = knn.predict_proba([test_example])
        best_n = np.argsort(predict_prob[0])[-5:]
        class_mapping = knn.classes_
        top_n_list = []
        for p in best_n:
            top_n_list.append(class_mapping[p])
        if class_label in top_n_list:
            count_correct = count_correct + 1

    acc_knn_top5 = float(count_correct) / float(len(X_test))
    acc_knn_top5 = float("{0:.15f}".format(round(acc_knn_top5, 6)))

    return acc_knn_top1, acc_knn_top2, acc_knn_top5


def AWF_Disjoint_Memory_Experment():
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with similar distribution.
    The model is trained on AWF777 and tested on AWF100 and the set of websites
    in the training set and the testing set are mutually exclusive.
    '''
    model_path = '../model_training/trained_model/Triplet_Model.h5'
    features_model = load_model(model_path)
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper
    SOP_list = [100]
    # SOP_list is the size of problem (how large the closed world is)
    # You can run gird search for various sizes of problems
    # SOP_list = [100, 75, 50, 25, 10]
    n_shot_list = [5]
    # n_shot_list is the number of n examples (n-shot)
    # You can run grid search for various sizes of n-shot
    # n_shot_list = [1, 5, 10, 15, 20]

    for size_of_problem in SOP_list:
        print("SOP:", size_of_problem)
        for n_shot in n_shot_list:
            acc_list_Top1 = []
            acc_list_Top2 = []
            acc_list_Top5 = []
            list_past_acc_top1 = []
            list_past_acc_top2 = []
            list_past_acc_top5 = []

            list_current_acc_top1 = []
            list_current_acc_top2 = []
            list_current_acc_top5 = []
            for i in range(10):
                list_past_acc_top1.append([[],[],[]])
                list_past_acc_top2.append([[],[],[]])
                list_past_acc_top5.append([[],[],[]])
    
                list_current_acc_top1 .append([[],[],[]])
                list_current_acc_top2 .append([[],[],[]])
                list_current_acc_top5 .append([[],[],[]])

            for i in range(10):
                signature_mev_vector_dict,signature_vector_dict, test_vector_dict = create_test_set_AWF_disjoint(features_model=features_model,
                                                                                       n_instance=90, max_n=20,
                                                                                       n_shot=n_shot)
                # Measure the performance (accuracy)
                func = [model1_knn_accuracy,model2_accuracy,model3_accuracy]

                #排障
                # past_acc_top1, past_acc_top2, past_acc_top5, current_acc_top1, current_acc_top2, current_acc_top5 = \
                #     func[2](signature_vector_dict, test_vector_dict, 10, n_shot, 10)


                for j in range(3):
                    if j != 2:
                        past_acc_top1, past_acc_top2, past_acc_top5, current_acc_top1, current_acc_top2, current_acc_top5 = \
                            func[j](signature_mev_vector_dict, test_vector_dict, 10, n_shot, 10)
                    else:
                        past_acc_top1, past_acc_top2, past_acc_top5, current_acc_top1, current_acc_top2, current_acc_top5 = \
                            func[j](signature_vector_dict, test_vector_dict, 10, n_shot, 10)

                    list_past_acc_top1[i][j].append(past_acc_top1)
                    list_past_acc_top2[i][j].append(past_acc_top2)
                    list_past_acc_top5[i][j].append(past_acc_top5)
                    
                    list_current_acc_top1[i][j].append(current_acc_top1)
                    list_current_acc_top2[i][j].append(current_acc_top2)
                    list_current_acc_top5[i][j].append(current_acc_top5)

            with open('result.txt','w') as f:
                f.write("N_shot:"+str(n_shot)+"\n")
                for mode in range(3):
                    f.write("\nmode"+str(mode+1)+"\n")
                    for epoch in range(10):
                        f.write("   epoch"+str(epoch+1)+"\n")
                        f.write("       past:\n")
                        f.write("           top1:")
                        f.write(str(list_past_acc_top1[epoch][mode]).strip('[]'))
                        f.write("\n")
                        f.write("           top2")
                        f.write(str(list_past_acc_top2[epoch][mode]).strip('[]'))
                        f.write("\n")
                        f.write("           top5")
                        f.write(str(list_past_acc_top5[epoch][mode]).strip('[]'))
                        f.write("\n")
                        f.write("       current:\n")
                        f.write("           top1:")
                        f.write(str(list_current_acc_top1[epoch][mode]).strip('[]'))
                        f.write("\n")
                        f.write("           top2")
                        f.write(str(list_current_acc_top2[epoch][mode]).strip('[]'))
                        f.write("\n")
                        f.write("           top5")
                        f.write(str(list_current_acc_top5[epoch][mode]).strip('[]'))



#todo 需要添加TTA的部分，大于某个置信度就加入原型并平均。
def model1_knn_accuracy(signature_vector_dict, test_vector_dict, size_of_problem, n_shot, epoch):

    X_train = []
    y_train = []

    list_past_acc_top1=[]
    list_past_acc_top2 = []
    list_past_acc_top5 = []
    list_current_acc_top1 =[]
    list_current_acc_top2 = []
    list_current_acc_top5 = []

    site_labels = list(signature_vector_dict.keys())
    random.shuffle(site_labels)
    for i in range(epoch):
        tested_sites= site_labels[i*size_of_problem:(i+1)*size_of_problem]
        for s in tested_sites:
            for each_test in range(len(signature_vector_dict[s])):
                X_train.append(signature_vector_dict[s][each_test])
                y_train.append(s)
        X_test_past=[]
        Y_test_past=[]
        tested_past_sites = site_labels[:i*size_of_problem]
        for s in tested_past_sites:
            for each_test in range(len(test_vector_dict[s])):
                X_test_past.append(test_vector_dict[s][each_test])
                Y_test_past.append(s)
        X_test_current=[]
        Y_test_current=[]
        tested_current_sites = site_labels[i*size_of_problem:(i+1)*size_of_problem]
        for s in tested_current_sites:
            for each_test in range(len(test_vector_dict[s])):
                X_test_current.append(test_vector_dict[s][each_test])
                Y_test_current.append(s)

        knn = KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')
        knn.fit(X_train,y_train)

        #top1
        if i!=0:
            past_acc_knn_top1=accuracy_score(Y_test_past,knn.predict(X_test_past))
            past_acc_knn_top1=float("{0:.15f}".format(round(past_acc_knn_top1,6)))
            list_past_acc_top1.append(past_acc_knn_top1)
        else:
            list_past_acc_top1.append(0)

        current_acc_knn_top1=accuracy_score(Y_test_current,knn.predict(X_test_current))
        current_acc_knn_top1=float("{0:.15f}".format(round(current_acc_knn_top1,6)))
        list_current_acc_top1.append(current_acc_knn_top1)

        #top2
        if i!=0:
            count_correct = 0
            for s in range(len(X_test_past)):
                test_example = X_test_past[s]
                class_label = Y_test_past[s]
                predict_prob = knn.predict_proba([test_example])
                best_n = np.argsort(predict_prob[0])[-2:]
                class_mapping = knn.classes_
                top_n_list = []
                for p in best_n:
                    top_n_list.append(class_mapping[p])
                if class_label in top_n_list:
                    count_correct = count_correct + 1
            past_acc_knn_top2=float(count_correct)/float(len(X_test_past))
            past_acc_knn_top2=float("{0:.15f}".format(round(past_acc_knn_top2,6)))
            list_past_acc_top2.append(past_acc_knn_top2)
        else:
            list_past_acc_top2.append(0)

        count_correct = 0
        for s in range(len(X_test_current)):
            test_example = X_test_current[s]
            class_label = Y_test_current[s]
            predict_prob = knn.predict_proba([test_example])
            best_n = np.argsort(predict_prob[0])[-2:]
            class_mapping = knn.classes_
            top_n_list = []
            for p in best_n:
                top_n_list.append(class_mapping[p])
            if class_label in top_n_list:
                count_correct = count_correct + 1
        current_acc_knn_top2 = float(count_correct) / float(len(X_test_current))
        current_acc_knn_top2 = float("{0:.15f}".format(round(current_acc_knn_top2, 6)))
        list_current_acc_top2.append(current_acc_knn_top2)

        #top5
        if i!=0:
            count_correct = 0
            for s in range(len(X_test_past)):
                test_example = X_test_past[s]
                class_label = Y_test_past[s]
                predict_prob = knn.predict_proba([test_example])
                best_n = np.argsort(predict_prob[0])[-5:]
                class_mapping = knn.classes_
                top_n_list = []
                for p in best_n:
                    top_n_list.append(class_mapping[p])
                if class_label in top_n_list:
                    count_correct = count_correct + 1
            past_acc_knn_top5 = float(count_correct) / float(len(X_test_past))
            past_acc_knn_top5 = float("{0:.15f}".format(round(past_acc_knn_top5, 6)))
            list_past_acc_top5.append(past_acc_knn_top5)
        else:
            list_past_acc_top5.append(0)

        count_correct = 0
        for s in range(len(X_test_current)):
            test_example = X_test_current[s]
            class_label = Y_test_current[s]
            predict_prob = knn.predict_proba([test_example])
            best_n = np.argsort(predict_prob[0])[-5:]
            class_mapping = knn.classes_
            top_n_list = []
            for p in best_n:
                top_n_list.append(class_mapping[p])
            if class_label in top_n_list:
                count_correct = count_correct + 1
        current_acc_knn_top5 = float(count_correct) / float(len(X_test_current))
        current_acc_knn_top5 = float("{0:.15f}".format(round(current_acc_knn_top5, 6)))
        list_current_acc_top5.append(current_acc_knn_top5)

        print("mode1 epoch:"+str(i))
    return list_past_acc_top1,list_past_acc_top2,list_past_acc_top5,list_current_acc_top1,list_current_acc_top2,list_current_acc_top5


def model2_accuracy(signature_vector_dict, test_vector_dict, size_of_problem,n_shot,epoch):
    X_prototype=[]
    Y_prototype=[]

    list_past_acc_top1 = []
    list_past_acc_top2 = []
    list_past_acc_top5 = []
    list_current_acc_top1 = []
    list_current_acc_top2 = []
    list_current_acc_top5 = []

    site_labels = list(signature_vector_dict.keys())
    random.shuffle(site_labels)
    for i in range(epoch):
        prototype_sites=site_labels[i*size_of_problem:(i+1)*size_of_problem]
        for s in prototype_sites:
            for each_test in range(len(signature_vector_dict[s])):
                X_prototype.append(signature_vector_dict[s][each_test])
                Y_prototype.append(s)

        X_test_past=[]
        Y_test_past=[]
        test_past_sites = site_labels[:i*size_of_problem]
        for s in test_past_sites:
            for each_test in range(len(test_vector_dict[s])):
                X_test_past.append(test_vector_dict[s][each_test])
                Y_test_past.append(s)

        X_test_current = []
        Y_test_current = []
        test_current_sites = site_labels[i*size_of_problem:(i+1) * size_of_problem]
        for s in test_current_sites:
            for each_test in range(len(test_vector_dict[s])):
                X_test_current.append(test_vector_dict[s][each_test])
                Y_test_current.append(s)

        Y_prototype_np = np.array(Y_prototype)
        if i!=0:
            cosine_similarity_past = cosine_similarity(X_test_past,X_prototype)
            past_best_1 = np.argsort(cosine_similarity_past,axis=1)[:,-1:].flatten()
            past_acc_top1=accuracy_score(Y_test_past,Y_prototype_np[past_best_1])
            past_acc_top1 = float("{0:.15f}".format(round(past_acc_top1, 6)))
            list_past_acc_top1.append(past_acc_top1)
        else:
            list_past_acc_top1.append(0)


        cosine_similarity_current = cosine_similarity(X_test_current,X_prototype)
        current_best_1 = np.argsort(cosine_similarity_current,axis=1)[:,-1:].flatten()
        current_acc_top1 = accuracy_score(Y_test_current,Y_prototype_np[current_best_1])
        current_acc_top1 = float("{0:.15f}".format(round(current_acc_top1, 6)))
        list_current_acc_top1.append(current_acc_top1)

        #top2
        if i!=0:
            count_correct = 0
            past_best_2 = np.argsort(cosine_similarity_past,axis=1)[:,-2:]
            for s in range(len(X_test_past)):
                class_label = Y_test_past[s]
                best_2 = past_best_2[s]
                top_2_list=Y_prototype_np[best_2]
                if class_label in top_2_list:
                    count_correct +=1
            past_acc_top2 = float(count_correct) / float(len(X_test_past))
            past_acc_top2 = float("{0:.15f}".format(round(past_acc_top2, 6)))
            list_past_acc_top2.append(past_acc_top2)
        else:
            list_past_acc_top2.append(0)

        count_correct = 0
        current_best_2 = np.argsort(cosine_similarity_current, axis=1)[:, -2:]
        for s in range(len(X_test_current)):
            class_label = Y_test_current[s]
            best_2 = current_best_2[s]
            top_2_list = Y_prototype_np[best_2]
            if class_label in top_2_list:
                count_correct += 1
        current_acc_top2 = float(count_correct) / float(len(X_test_current))
        current_acc_top2 = float("{0:.15f}".format(round(current_acc_top2, 6)))
        list_current_acc_top2.append(current_acc_top2)

        #top5
        if i!=0:
            count_correct = 0
            past_best_5 = np.argsort(cosine_similarity_past, axis=1)[:, -5:]
            for s in range(len(X_test_past)):
                class_label = Y_test_past[s]
                best_5 = past_best_5[s]
                top_5_list = Y_prototype_np[best_5]
                if class_label in top_5_list:
                    count_correct += 1
            past_acc_top5 = float(count_correct) / float(len(X_test_past))
            past_acc_top5 = float("{0:.15f}".format(round(past_acc_top5, 6)))
            list_past_acc_top5.append(past_acc_top5)
        else:
            list_past_acc_top5.append(0)

        count_correct = 0
        current_best_5 = np.argsort(cosine_similarity_current, axis=1)[:, -5:]
        for s in range(len(X_test_current)):
            class_label = Y_test_current[s]
            best_5 = current_best_5[s]
            top_5_list = Y_prototype_np[best_5]
            if class_label in top_5_list:
                count_correct += 1
        current_acc_top5 = float(count_correct) / float(len(X_test_current))
        current_acc_top5 = float("{0:.15f}".format(round(current_acc_top5, 6)))
        list_current_acc_top5.append(current_acc_top5)

        print("mode2 epoch:" + str(i))
    return list_past_acc_top1,list_past_acc_top2,list_past_acc_top5,list_current_acc_top1,list_current_acc_top2,list_current_acc_top5


def model3_accuracy(signature_vector_dict, test_vector_dict, size_of_problem, n_shot, epoch):
    X_prototype = []
    Y_prototype = []

    list_past_acc_top1 = []
    list_past_acc_top2 = []
    list_past_acc_top5 = []
    list_current_acc_top1 = []
    list_current_acc_top2 = []
    list_current_acc_top5 = []

    site_labels = list(signature_vector_dict.keys())
    random.shuffle(site_labels)
    for i in range(epoch):
        prototype_sites = site_labels[i * size_of_problem:(i + 1) * size_of_problem]
        for s in prototype_sites:
            for each_test in range(len(signature_vector_dict[s])):
                X_prototype.append(signature_vector_dict[s][each_test])
                Y_prototype.append(s)
        X_test_past = []
        Y_test_past = []
        test_past_sites = site_labels[:i * size_of_problem]
        for s in test_past_sites:
            for each_test in range(len(test_vector_dict[s])):
                X_test_past.append(test_vector_dict[s][each_test])
                Y_test_past.append(s)

        X_test_current = []
        Y_test_current = []
        test_current_sites = site_labels[i * size_of_problem:(i + 1) * size_of_problem]
        for s in test_current_sites:
            for each_test in range(len(test_vector_dict[s])):
                X_test_current.append(test_vector_dict[s][each_test])
                Y_test_current.append(s)

        Y_prototype_np = np.array(Y_prototype)
        if i!=0:
            cosine_similarity_past = cosine_similarity(X_test_past, X_prototype)
            past_best_n_shot = np.argsort(cosine_similarity_past, axis=1)[:, -n_shot:]

        cosine_similarity_current = cosine_similarity(X_test_current,X_prototype)
        current_best_n_shot = np.argsort(cosine_similarity_current,axis=1)[:,-n_shot:]

        if i!=0:
            count_correct_past_top1 = 0
            count_correct_past_top2 = 0
            count_correct_past_top5 = 0
            for s in range(len(X_test_past)):
                class_label = Y_test_past[s]
                past_best = past_best_n_shot[s]
                top_n_list = Y_prototype_np[past_best]
                top_n_dict = Counter(top_n_list)
                if class_label in list(top_n_dict.keys())[:1]:
                    count_correct_past_top1+=1
                if class_label in list(top_n_dict.keys())[:2]:
                    count_correct_past_top2+=1
                if class_label in list(top_n_dict.keys())[:5]:
                    count_correct_past_top5+=1
            past_acc_top1 = float(count_correct_past_top1)/float(len(X_test_past))
            past_acc_top2 = float(count_correct_past_top2) / float(len(X_test_past))
            past_acc_top5 = float(count_correct_past_top5) / float(len(X_test_past))
            past_acc_top1 = float("{0:.15f}".format(round(past_acc_top1, 6)))
            past_acc_top2 = float("{0:.15f}".format(round(past_acc_top2, 6)))
            past_acc_top5 = float("{0:.15f}".format(round(past_acc_top5, 6)))
            list_past_acc_top1.append(past_acc_top1)
            list_past_acc_top2.append(past_acc_top2)
            list_past_acc_top5.append(past_acc_top5)
        else:
            list_past_acc_top1.append(0)
            list_past_acc_top2.append(0)
            list_past_acc_top5.append(0)

        count_correct_current_top1 = 0
        count_correct_current_top2 = 0
        count_correct_current_top5 = 0
        for s in range(len(X_test_current)):
            class_label = Y_test_current[s]
            current_best = current_best_n_shot[s]
            top_n_list = Y_prototype_np[current_best]
            top_n_dict = Counter(top_n_list)
            if class_label in list(top_n_dict.keys())[:1]:
                count_correct_current_top1 += 1
            if class_label in list(top_n_dict.keys())[:2]:
                count_correct_current_top2 += 1
            if class_label in list(top_n_dict.keys())[:5]:
                count_correct_current_top5 += 1
        current_acc_top1 = float(count_correct_current_top1) / float(len(X_test_current))
        current_acc_top2 = float(count_correct_current_top2) / float(len(X_test_current))
        current_acc_top5 = float(count_correct_current_top5) / float(len(X_test_current))
        current_acc_top1 = float("{0:.15f}".format(round(current_acc_top1, 6)))
        current_acc_top2 = float("{0:.15f}".format(round(current_acc_top2, 6)))
        current_acc_top5 = float("{0:.15f}".format(round(current_acc_top5, 6)))
        list_current_acc_top1.append(current_acc_top1)
        list_current_acc_top2.append(current_acc_top2)
        list_current_acc_top5.append(current_acc_top5)
        print("mode3 epoch:" + str(i))
    return list_past_acc_top1, list_past_acc_top2, list_past_acc_top5, list_current_acc_top1, list_current_acc_top2, list_current_acc_top5


if __name__ == '__main__':
    AWF_Disjoint_Memory_Experment()