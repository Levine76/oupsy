#import tensorflow as tf
import numpy as np
import sys
import os
import pickle
import random
import glob
import re
import math

def create_training_sequence_point_wise_for_AE(data_dir, lable_dir, save_dir):
    
    files1 = glob.glob(os.path.join(data_dir, "*.ply"))
    files2 = glob.glob(os.path.join(lable_dir, "*.ply"))
    
    files1.sort()   
    files2.sort()   
        
    if len(files1) != len(files2): 
        print('length do not match')
        return 
    for i in range(len(files2)): 
        print(save_dir + files1[i].split('.')[0]+'--'+files2[i].split('.')[0]+".obj")
#         exit(0)
        file_dump = open( files1[i].split('.')[0]+'--'+files2[i].split('.')[0]+".obj",'wb')
        pickle.dump([files1[i],files2[i]], file_dump)
        file_dump.close()
  


# create training sequence list wise for RNN
def create_training_sequence_list_wise_for_RNN(data_dir,lable_dir,save_dir):

    files1 = glob.glob(os.path.join(data_dir, "*.ply"))
    files2 = glob.glob(os.path.join(lable_dir, "*.ply"))
    

    files1.sort()   
    files2.sort()   
    
    
    start_index = 0
    end_index = 0
    index = []
    for item in files2:
        digital_str_list = re.findall(r"\d",item)
        digital = int(''.join(digital_str_list))
        index.append(digital)
    
    index.sort()
    if index[-1] < len(files1):
        index.append(len(files1))

    print(index)
    for i in range(len(files2)-1):
        
        file_dump = open(save_dir+'/'+'RawPC_'+ str(index[i])+'_'+str(index[i+1])+'-'+'TrunkPC-'+str(index[i]) +".obj",'wb')
        print(files1[index[i]:index[i+1]],item[i])
        pickle.dump([files1[index[i]:index[i+1]],files2[i]], file_dump)
        file_dump.close()



def get_data(testPart,data_dir, is_return_dir = False) :
    
    files = glob.glob(os.path.join(data_dir, "*.obj"))
    random.shuffle(files)
      
    sep = int(len(files)*(1-testPart))
    trainFiles = files[0:sep]
    testFiles = files[sep:]

    x_trains = []
    x_tests = []
    y_trains = []
    y_tests = []

    count_train = 0

    for file in trainFiles :
        
        f = open(file,'rb')
        obj = pickle.load(f,encoding='bytes')
        f.close()
        #print(file)
        #print(len(obj))
        #print(obj)
        #print(len(obj[0]),len(obj[0][0]),len(obj[1]))
        #exit(0)
        if is_return_dir:
            x_trains.append(obj[0])
        else:
            x_trains.append(np.array(obj[0][0],dtype = 'float16') )
            
        if is_return_dir:
            y_trains.append(obj[1])
        else:
            y_trains.append(np.array(obj[1],dtype = 'float16'))
        count_train += 1
        print("train data :",count_train)
    
    
    count_test = 0
    for file in testFiles :
        f = open(file,'rb')
        obj = pickle.load(f,encoding='bytes')
        f.close()
     
        #print(len(obj[0]),len(obj[0][0]),len(obj[1]))
        if is_return_dir:
            x_tests.append(obj[0])
        else:
            x_tests.append(np.asarray(obj[0],dtype = 'float16') )
        
        if is_return_dir:
            y_tests.append(obj[1])
        else:
            y_tests.append(np.asarray(obj[1],dtype = 'float16'))
            
        count_test+=1
        print("test data :",count_test)


    return x_trains, y_trains, x_tests, y_tests


def get_ae_data(testPart,data_dir, is_return_dir = False) :
    
    files = glob.glob(os.path.join(data_dir, "*.ply"))
    random.shuffle(files)
    
    print(len(files))
    sep = int(len(files)*(1-testPart))
    trainFiles = files[0:sep]
    testFiles = files[sep:]

    x_trains = []
    x_tests = []
    count_train = 0
    
    for file in trainFiles :
        x_trains.append(file)
        count_train += 1
        print("train data :",count_train)
    
    count_test = 0
    
    for file in testFiles :
        x_tests.append(file)
        
        count_test+=1
        print("test data :",count_test)
        
    return x_trains, x_tests



def transform(point_cloud_vertices,is_full_size = False) :
    if is_full_size:
        x_train_transformed = np.zeros((1,1024,1024,1024),dtype = 'float16')
    else:
        x_train_transformed = np.zeros((1,512,512,512),dtype = 'float16')
    
    #print(point_cloud_vertices)
    point_cloud_vertices[point_cloud_vertices<0] = -point_cloud_vertices[point_cloud_vertices<0]
    #print(point_cloud_vertices)
    #exit(0)
    for pos in point_cloud_vertices :
        
        if not is_full_size:
            if 256 <= pos[0]<768  and 256 <= pos[1]<768 and 256 <= pos[2]< 768 :
                x_train_transformed[0,int(pos[0]-256),int(pos[1]-256),int(pos[2]-256)] = 1
        else:
            x_train_transformed[0,int(pos[0]),int(pos[1]),int(pos[2])] = 1
    return x_train_transformed


def transform_scaling(point_cloud_vertices,Original_dimension=1024,scale_factor=0.2) :
   
    new_dimension = int(math.ceil(scale_factor*Original_dimension))
    x_train_transformed = np.zeros((1,new_dimension,new_dimension,new_dimension),dtype = 'float16')

    point_cloud_vertices[point_cloud_vertices<0] = -point_cloud_vertices[point_cloud_vertices<0]
    for pos in point_cloud_vertices :
        x_train_transformed[0,int(math.ceil(pos[0]*scale_factor)),
                            int(math.ceil(pos[1]*scale_factor)),int(math.ceil(pos[2]*scale_factor))] = 1
    return x_train_transformed

def tensor_to_ply(tensor,threshold,save_dir,scale_factor=None,Original_dimension=1024):
    print('start to write tensor to ply')
    point_counts = len(tensor[tensor > threshold])
    head = "ply\n"+\
        "format ascii 1.0\n"+\
        "comment written by quantization_proc\n"+\
        "element vertex "+ str(point_counts) + "\n"+\
        "property float32 x\n"+\
        "property float32 y\n"+\
        "property float32 z\n"+\
        "end_header\n"

    body = ""
    if scale_factor:
        for i in range(len(tensor)):
            for j in range(len(tensor[0])):
                for k in range(len(tensor[0][0])):
                    if tensor[i][j][k] >= threshold:
                        body += ' '.join([str(min(int(i/scale_factor),1023)),
                                          str(min(int(j/scale_factor),1023)),
                                          str(min(int(k/scale_factor),1023))]) + "\n"
    else:
        for i in range(len(tensor)):
            for j in range(len(tensor[0])):
                for k in range(len(tensor[0][0])):
                    if tensor[i][j][k] >= threshold:
                        body += ' '.join([str(i),str(j),str(k)]) + "\n"
    
    file = open(save_dir, "w+")
    file.write(head)
    file.write(body)
    file.close()



if __name__ == '__main__':
    """
    raw_ply = '../data/longdress/Ply/'
    compressed_ply = '../data/longdress/Compressed_Ply/'
    save_dir = '../data/longdress/training_dir_point_wise/'
    create_training_sequence_point_wise_for_AE(raw_ply,compressed_ply,save_dir)
    """

    raw_ply = '../data/longdress2/Ply/'
    lable_dir = '../data/longdress2/trunk_ply/'
    save_dir = '../data/longdress2/rnn_train_sequence/'
    create_training_sequence_list_wise_for_RNN(raw_ply,lable_dir,save_dir)

