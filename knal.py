ƒimport math
import operator
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import *
from collections import Counter
import random
from matplotlib import pyplot as p
import sys

#Method 1 - printing progress bar
def progress(count, iterations, suffix=''):
    bar_length = 60
    filled_bar_length = int(round(bar_length * count / (iterations)))

    percentage = round(100.0 * count / (iterations), 1)
    progress_bar = '=' * filled_bar_length + '-' * (bar_length - filled_bar_length)

    sys.stdout.write('[%s] %s%s ...%s\r' % (progress_bar, percentage, '%', suffix))
    sys.stdout.flush() 
    
#Method 2- getting labels of reviews
def get_labels(file_name):
    labels = {}
    label_indx = 0
    with open(file_name) as file:
        for line in file:
            label= line.strip().split()[0]
            labels[label_indx]= label
            label_indx += 1
    return labels

#Method 3- getting the feature sparse matrix and the whole feature set
def get_features(file_name):
    features = {}
    data = []
    indptr = [0]
    indices = []
    with open(file_name) as file:
        for line in file:
            line = line.strip().split()[1:]
            for w in line:
                index = features.setdefault(w, len(features))
                indices.append(index)
                data.append(1)
            
            indptr.append(len(indices))

    feature_vectors = csr_matrix((data, indices, indptr), dtype=float)

    return features, feature_vectors

#Metod 4- converting 408 reviews to an array form and L2 normalising them
def l2_norm_feats(feat_vects):
    
    reviews = len(feat_vects.getnnz(axis=1))
    l2 = []
    for i in range(reviews):
        array = feat_vects[i].toarray()
        array = array/np.sqrt(array.sum())
        l2.append(array)
    
    return l2

#Method 5- clustering, standard k-means (means as centroids)
def mean_as_centroid(k, data):
    
    #creating k initial centroids
    centroids = {}
    for i in range(k):
        centroids[i] = random.choice(data)
    
    #iterations
    for itera in range(300):
        #dict storing vectors representing reviews for all the k clusters
        clusters_rev_vects = {}
        #dict storing indices representing reviews for all the k clusters
        clusters_indices = {}

        for i in range(k):
            #lists storing vectors representing reviews
            clusters_rev_vects[i] = []
            #lists storing indices of reviews in each cluster
            clusters_indices[i] = []
        
        """assigning data points to clusters, aim is to minimise the 
        distance between data points and cluster centers (WCSS)"""
        for index, review in enumerate(data):
            distances = [np.sqrt(np.sum((review - centroids[centroid])**2)) for centroid in centroids]
            classification = distances.index(min(distances))
            clusters_rev_vects[classification].append(review)
            clusters_indices[classification].append(index)
    
        #dictionary holding all the centroids generated at the beginning
        prev_centroids = dict(centroids)

        #calculating mean of each cluster, these will be the new updated centroids
        for review in clusters_rev_vects:
            centroids[review] = np.average(clusters_rev_vects[review], axis=0)

        """convergence assumptions: minimizing the WCSS, if the distance 
        calculated below is 0, data points do not move between the clusters;
        alternatively np.array_equal() could be used to assess if the data
        points do not move"""
        optimized = True
        
        for c in centroids:
            original_centroid = prev_centroids[c]
            current_centroid = centroids[c]
            y = np.sqrt(np.sum((current_centroid-original_centroid)**2))
            if y > 0.0001:
                optimized = False
        
        if optimized:
            print ('\nConvergance achieved in %d iterations' % (itera+1))
            break
            
    return clusters_indices
    
#Method 6- clustering, instance closest to the mean as centroid
def instance_as_centroid(k, data):
    
    #creating k initial centroids
    centroids = {}
    for i in range(k):
        centroids[i] = random.choice(data)
        
    #iterations
    for itera in range(300):
        #dict storing vectors representing reviews for all the clusters
        clusters_rev_vects = {}
        #dict storing indices representing reviews for all the clusters
        clusters_indices = {}

        for i in range(k):
            #lists storing vectors representing reviews
            clusters_rev_vects[i] = []
            #lists storing indices of reviews in each cluster
            clusters_indices[i] = []
        
        """assigning data points to cluster, aim is to minimise the 
        distance between data points and cluster centers (WCSS)"""
        for index, review in enumerate(data):
            distances = [np.sqrt(np.sum((review - centroids[centroid])**2)) for centroid in centroids]
            classification = distances.index(min(distances))
            clusters_rev_vects[classification].append(review)
            clusters_indices[classification].append(index)
    
        # dictionary holding all the centroids generated at the beginning of each iteration
        prev_centroids = dict(centroids)
    
        #calculating centroids- means
        for review in clusters_rev_vects:
            centroids[review] = np.average(clusters_rev_vects[review], axis=0)
        
        #updating centroids- calculating instances closest to the mean
        dista = []
        for label, cluster in clusters_indices.items():
            d = []
            for i in range(len(cluster)):
                d_istance = np.sqrt(np.sum((data[cluster[i]] - centroids[label])**2))
                d.append(d_istance)
            dista.append(d)
        
        #updating centroids - instances closest to the mean will be k centroids
        for index, i in enumerate(dista):
            if len(i) > 0:
                min_index = np.argmin(i)
                centroids[index] = data[min_index]

        """convergence assumptions: minimizing the WCSS, in this case the
        distance between data points and clusters is never minimised, 
        data points keep moving between the clusters- see the .pdf for explanation"""
        optimized = True

        for c in centroids:
            original_centroid = prev_centroids[c]
            current_centroid = centroids[c]
            y = np.sqrt(np.sum((current_centroid-original_centroid)**2))
            if y > 0.0001:
                optimized = False

        #this version of algorithm never optimises, thus it is terminated after 10 iterations
        if optimized or itera == 10:
            break
    
    return clusters_indices

#Method 7- labelling elements of each cluster as the methods above return only indices
def label_elements(clusters_indices, feats_labels):
    
    evaluation_dict = {}
    for label, values in clusters_indices.items():
        temp = {}
        for value in values:
            if feats_labels[value] not in temp:
                temp[feats_labels[value]] = 1
            else:
                temp[feats_labels[value]] += 1
        
        evaluation_dict[label] = temp
    
    return evaluation_dict

#Method 8- merging dictionaries using Counter (from collections)
def counter(dict1, dict2):
    
    for i in (dict1,dict2):
        dict1 = Counter(dict1)
        dict2 = Counter(dict2)
        merged = dict1 + dict2
    
    return dict(merged)

#Method 9- merging clusters with the same highest key value as label
def merging_clusters(clusters_labelled_elements):
    
    #list holding keys with max value for each cluster
    l = []
    for i in clusters_labelled_elements:
        if len(clusters_labelled_elements[i]) > 0:
            maxx = max(clusters_labelled_elements[i].items(), key=operator.itemgetter(1))[0]
        l.append(maxx)
    
    #Merging
    clusters = {}
    tags= {}
    for index, tag in enumerate(l):
        if tag not in tags:
            tags[tag] = index
            clusters[index]= clusters_labelled_elements[index]
        else:
            clusters[tags[tag]] = counter(clusters[tags[tag]], clusters_labelled_elements[index])
    
    return clusters
    
#Method 10- labelling merged clusters
def labelling_merged_clusters(clusters_merged):
    
    l_merged_clusters = {}
    
    for label, items in clusters_merged.items():
        x = max(items, key=items.get)
        l_merged_clusters[x] = items
    
    return l_merged_clusters

#Method 11- calculating macro-averaged recall for clusters with varied k
def macro_avg_recall(clusters_merged_labelled):
    
    recall = []
    for label, items in clusters_merged_labelled.items():
        r = []
        for label, item in items.items():
            r.append(item)
        recall.append(r)
    
    macro_recall = []
    for i in recall:
        rec = (max(i)/51)*100
        macro_recall.append(rec)
    
    macro_recall = sum(macro_recall)/len(macro_recall)
    
    return macro_recall

#Method 12- calculating macro-averaged precision for clusters with varied k
def macro_avg_precision(clusters_merged_labelled):
    
    precision = []
    for label, items in clusters_merged_labelled.items():
        p = []
        for label, item in items.items():
            p.append(item)
        precision.append(p)
    
    macro_precision = []
    for i in precision:
        pre = (max(i)/sum(i))*100
        macro_precision.append(pre)
        
    
    macro_precision = sum(macro_precision)/len(macro_precision)
    
    return macro_precision

#Method 13- calculating macro-averaged precision for clusters with varied k
def macro_avg_f_score(clusters_ma_recall, clusters_ma_precision):
    
    f = (2 * clusters_ma_precision * clusters_ma_recall) / (clusters_ma_precision + clusters_ma_recall)
    
    return f

#Method 14- MAIN
if __name__ == '__main__':
    
    #Getting labels, all features, and feature vectors
    feats_labels = get_labels('data.txt') 
    feats, feat_vects = get_features('data.txt')
    
    #L2 normalisation of feature vetors and extracting arrays from the sparse matrix
    l2feat_vects = l2_norm_feats(feat_vects)
    
    #Creating lists to hold Marco-Averaged Recall, Precision and F-score for all cluster the the range(2,21)
    global_macro_averaged_Recall = []
    global_macro_averaged_Precision = []
    global_macro_averaged_F_Score = []
    
    #Execution
    print ('\n****   Text Clustering using K-Means with the value of K from 2 to 20   ****')
    print ('\n1. Run K-Means')
    print ('2. Run the algorithm using closest instance to mean as centroid')
    user_input = int(input('\nPlease choose either 1 or 2 based on the above options: '))
    
    
    #Varying the K
    r = range(2,21)
    if user_input == 1:
        for k in r:
            print ('\n******  Clustering with K = %d  ******' % (k))
            progress(k, 20, suffix='')
            #1
            clusters_indices = mean_as_centroid(k, l2feat_vects)
            #2
            clusters_labelled_elements = label_elements(clusters_indices, feats_labels)
            #3
            clusters_merged = merging_clusters(clusters_labelled_elements)
            #4
            clusters_merged_labelled = labelling_merged_clusters(clusters_merged)
            #5
            clusters_ma_recall = macro_avg_recall(clusters_merged_labelled)
            global_macro_averaged_Recall.append(clusters_ma_recall)
            #6
            clusters_ma_precision = macro_avg_precision(clusters_merged_labelled)
            global_macro_averaged_Precision.append(clusters_ma_precision)
            #7
            clusters_ma_f_score = macro_avg_f_score(clusters_ma_recall, clusters_ma_precision)
            global_macro_averaged_F_Score.append(clusters_ma_f_score)
            print ('\nMacro-Averaged Recall is: %.2f%%' % (clusters_ma_recall))
            print ('Macro-Averaged Precision is: %.2f%%' % (clusters_ma_precision))
            print ('Macro-Averaged F-score is: %.2f%%\n' % (clusters_ma_f_score))
            print ('Cluster labels are: ')
            for index, lab in enumerate(clusters_merged_labelled):
                print ((index+1), lab)
        #graphs
        p.figure(figsize=(11,6))
        p.subplot(2, 2, 1)
        p.plot(r, global_macro_averaged_Recall, '--g')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged Recal l%')
        p.tight_layout()
        
        p.subplot(2, 2, 2)
        p.plot(r, global_macro_averaged_Precision, '--r')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged Precision %')
        p.tight_layout()
        
        p.subplot(2, 2, 3)
        p.plot(r, global_macro_averaged_F_Score, '--b')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged F-Score%')
        p.tight_layout()
        
        p.subplot(2, 2, 4)
        p.plot(r, global_macro_averaged_Recall, '--g', label='Recall')
        p.plot(r, global_macro_averaged_Precision, '--r', label='Precision')
        p.plot(r, global_macro_averaged_F_Score, '--b', label='F-Score')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged Measures %')
        p.legend(loc='upper center', prop={'size':6}, ncol=3)
        p.tight_layout()
        p.show()
    
    elif user_input ==2:
        for k in r:
            print ('\n******  Clustering with K = %d  ******' % (k))
            progress(k, 20, suffix='')
            #1
            clusters_indices = instance_as_centroid(k, l2feat_vects)
            #2
            clusters_labelled_elements = label_elements(clusters_indices, feats_labels)
            #3
            clusters_merged = merging_clusters(clusters_labelled_elements)
            #4
            clusters_merged_labelled = labelling_merged_clusters(clusters_merged)
            #5
            clusters_ma_recall = macro_avg_recall(clusters_merged_labelled)
            global_macro_averaged_Recall.append(clusters_ma_recall)
            #6
            clusters_ma_precision = macro_avg_precision(clusters_merged_labelled)
            global_macro_averaged_Precision.append(clusters_ma_precision)
            #7
            clusters_ma_f_score = macro_avg_f_score(clusters_ma_recall, clusters_ma_precision)
            global_macro_averaged_F_Score.append(clusters_ma_f_score)
            print ('\nMacro-Averaged Recall is: %.2f%%' % (clusters_ma_recall))
            print ('Macro-Averaged Precision is: %.2f%%' % (clusters_ma_precision))
            print ('Macro-Averaged F-score is: %.2f%%\n' % (clusters_ma_f_score))
            print ('Cluster labels are: ')
            for index, lab in enumerate(clusters_merged_labelled):
                print ((index+1), lab)

        #graphs
        p.figure(figsize=(11,6))
        p.subplot(2, 2, 1)
        p.plot(r, global_macro_averaged_Recall, '--g')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged Recal l%')
        p.tight_layout()
        
        p.subplot(2, 2, 2)
        p.plot(r, global_macro_averaged_Precision, '--r')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged Precision %')
        p.tight_layout()
        
        p.subplot(2, 2, 3)
        p.plot(r, global_macro_averaged_F_Score, '--b')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged F-Score%')
        p.tight_layout()
        
        p.subplot(2, 2, 4)
        p.plot(r, global_macro_averaged_Recall, '--g', label='Recall')
        p.plot(r, global_macro_averaged_Precision, '--r', label='Precision')
        p.plot(r, global_macro_averaged_F_Score, '--b', label='F-Score')
        p.axis('tight')
        p.xticks(r)
        p.xlabel('Number of Clusters')
        p.ylabel('Macro-Averaged Measures %')
        p.legend(loc='upper center', prop={'size':6}, ncol=3)
        p.tight_layout()
        p.show()
