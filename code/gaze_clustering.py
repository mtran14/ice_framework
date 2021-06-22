#!/home/mtran/anaconda3/bin/python
import pandas as pd
import os
import numpy as np
import math
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.preprocessing import scale

def best_eps(cluster_data, num_clusters):
    #min_samples = int(np.round(np.log(cluster_data.shape[0])))

    min_samples = cluster_data.shape[0] / 100 # the size of a cluster is determined by this
                        # (100/10000) = 1% at least to be considered a cluster
    #min_samples = 9
    eps_range = np.linspace(0.001, 1, num=50)
    prev = None

    for i in reversed(range(0, eps_range.shape[0])):
        current_eps = eps_range[i]
        db = DBSCAN(eps=current_eps, min_samples=min_samples).fit(cluster_data)
        labels = db.labels_
        cnt = (Counter(labels))
        ratio = cnt.most_common(1)[-1][1] / cnt.most_common(2)[-1][1]
        if(len(set(labels)) >= num_clusters and ratio < 10):
            if(prev == None):
                return current_eps, min_samples
            else:
                return current_eps , min_samples
        prev = current_eps
    default_val = 0.001
    return default_val, min_samples


def gaze_clustering(input_path, output_path, scale=False):
    for file in os.listdir(input_path):
        df = pd.read_csv(os.path.join(input_path, file))
        headers = list(df.columns)
        headers = [x.strip() for x in headers]
        file_data = df.values

        keep_col = [False for _ in range(file_data.shape[1])]
        keep_col[headers.index('gaze_angle_x')] = True
        keep_col[headers.index('gaze_angle_y')] = True
        gaze_data = file_data[:, keep_col]

        if(scale_bool):
            gaze_data = scale(gaze_data)

        num_clusters = 2 #2 people speaking
        eps_val, min_samples = best_eps(gaze_data, num_clusters)
        db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(gaze_data)

        labels = db.labels_
        labels = list(labels)
        most_common,num_most_common = Counter(labels).most_common(1)[0]
        cluster_count = {x:labels.count(x) for x in labels}
        contact_cluster = max(cluster_count, key=cluster_count.get)
        #============================================================#
        # draw the bounding box
        x_list = []  #store x axis of contact cluster
        y_list = []  #store y axis of contact cluster

        for i in range(0, len(labels)):
            if(labels[i] == contact_cluster):
                x_list.append(gaze_data[i][0])
                y_list.append(gaze_data[i][1])

        upper_bound_x = max(x_list)  #right-most angle
        lower_bound_x = min(x_list)  #left-most angle
        upper_bound_y = max(y_list)  #lowest angle
        lower_bound_y = min(y_list)  #highest angle

        #============================================================#
        contact_list = []
        contact_list_1 = []
        contact_list_2 = []
        contact_list_3 = []
        contact_list_4 = []
        contact_list_5 = []
        contact_list_6 = []
        contact_list_7 = []
        contact_list_8 = []
        contact_list_9 = []

        #all_data = df.values[:,[df.values.shape[1]-2, df.values.shape[1]-1]]
        # all_data = current_data

        for frame in gaze_data:
            if(frame[0] < lower_bound_x  and frame[1] < lower_bound_y):
                contact = 1
                contact_list_1.append(1)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(0)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(1)
            elif(frame[1] < lower_bound_y and upper_bound_x > frame[0] > lower_bound_x):
                contact = 2
                contact_list_1.append(0)
                contact_list_2.append(1)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(0)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(2)
            elif(frame[1] < lower_bound_y and frame[0] > upper_bound_x):
                contact = 3
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(1)
                contact_list_4.append(0)
                contact_list_5.append(0)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(3)
            elif(upper_bound_y > frame[1] > lower_bound_y and frame[0] < lower_bound_x):
                contact = 4
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(1)
                contact_list_5.append(0)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(4)
            elif(upper_bound_y > frame[1] > lower_bound_y and upper_bound_x > frame[0] > lower_bound_x):
                contact = 5
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(1)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(5)
            elif(upper_bound_y > frame[1] > lower_bound_y and frame[0] > upper_bound_x):
                contact = 6
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(0)
                contact_list_6.append(1)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(6)
            elif(frame[1] > upper_bound_y and frame[0] < lower_bound_x):
                contact = 7
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(0)
                contact_list_6.append(0)
                contact_list_7.append(1)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(7)
            elif(frame[1] > upper_bound_y and upper_bound_x > frame[0] > lower_bound_x):
                contact = 8
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(0)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(1)
                contact_list_9.append(0)
                contact_list.append(8)
            elif(frame[1] > upper_bound_y and frame[0] > upper_bound_x):
                contact = 9
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(0)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(1)
                contact_list.append(9)
            else:
                contact = 5
                contact_list_1.append(0)
                contact_list_2.append(0)
                contact_list_3.append(0)
                contact_list_4.append(0)
                contact_list_5.append(1)
                contact_list_6.append(0)
                contact_list_7.append(0)
                contact_list_8.append(0)
                contact_list_9.append(0)
                contact_list.append(5)

        file_data = df.values
        gaze_df = pd.DataFrame({'Region 1':contact_list_1, 'Region 2':contact_list_2, 'Region 3':contact_list_3, \
                            'Region 4':contact_list_4, 'Region 5':contact_list_5, 'Region 6':contact_list_6, \
                            'Region 7':contact_list_7, 'Region 8':contact_list_8, 'Region 9':contact_list_9, 'Region':contact_list})
        new_data = np.concatenate((file_data, gaze_df.values), axis = 1)
        #new_data = contact_list
        output = pd.DataFrame(new_data)
        header_list = list(df.columns) + list(gaze_df.columns)

        output.to_csv(os.path.join(output_path, file.split('.')[0]+'.csv'), header=header_list, index=False)
        print("Finish extracting !!!")

if __name__ == '__main__':
    input_path = '../data/openface_all/'
    output_path = '../data/gaze_output/'
    scale_bool = False
    gaze_clustering(input_path, output_path, scale_bool)
