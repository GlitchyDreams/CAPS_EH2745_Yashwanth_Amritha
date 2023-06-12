
"""
@author1: yashwanth Damodaran
@author1: amritha jayan
"""

import os
import math
import numpy as np
import pandas as pd

import random
import pandapower as pp
import matplotlib.pyplot as plt
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl


random.seed(1)
"""
Setting the seed value is particularly useful when you need to generate 
random numbers for experimentation, testing, or debugging purposes and want to 
ensure that the same sequence of random numbers is generated each time.
"""
#print(random.random())

print("This code runs the 4 scenarios given, creates timeseries data for training and testing")
def CAPS_test_net():
    
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init_vm_pu = "flat", init_va_degree = "dc", calculate_voltage_angles=True)
        
    b1 = pp.create_bus(net, 110, name="CLARK")
    b2 = pp.create_bus(net, 110, name="AMHERST")
    b3 = pp.create_bus(net, 110, name="WINLOCK")
    b4 = pp.create_bus(net, 110, name="BOWMAN")
    b5 = pp.create_bus(net, 110, name="TROY")
    b6 = pp.create_bus(net, 110, name="MAPLE")
    b7 = pp.create_bus(net, 110, name="GRAND")
    b8 = pp.create_bus(net, 110, name="WAUTAGA")
    b9 = pp.create_bus(net, 110, name="CROSS")
 
    
    pp.create_ext_grid(net, b1)
    l14 = pp.create_line(net, b1, b4, 10, "149-AL1/24-ST1A 110.0")
    l49 = pp.create_line(net, b4, b9, 10, "149-AL1/24-ST1A 110.0")
    l98 = pp.create_line(net, b9, b8, 10, "149-AL1/24-ST1A 110.0")
    l82 = pp.create_line(net, b8, b2, 10, "149-AL1/24-ST1A 110.0")
    l87 = pp.create_line(net, b8, b7, 10, "149-AL1/24-ST1A 110.0")
    l76 = pp.create_line(net, b7, b6, 10, "149-AL1/24-ST1A 110.0")
    l63 = pp.create_line(net, b6, b3, 10, "149-AL1/24-ST1A 110.0")
    l65 = pp.create_line(net, b6, b5, 10, "149-AL1/24-ST1A 110.0")
    l54 = pp.create_line(net, b5, b4, 10, "149-AL1/24-ST1A 110.0")
    
    pp.create.create_switch(net, b1, l14, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b4, l49, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b9, l98, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b8, l82, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b8, l87, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b7, l76, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b6, l63, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b5, l65, et="b", closed=True, type="DS")
    pp.create.create_switch(net, b5, l54, et="b", closed=True, type="DS")
    

    pp.create_load(net, b5, p_mw=90., q_mvar=30., name='load5')
    pp.create_load(net, b7, p_mw=100., q_mvar=35., name='load7')
    pp.create_load(net, b9, p_mw=125., q_mvar=50., name='load9')
    
    pp.create_sgen(net, b1, p_mw=0., q_mvar=0., name='sgen1')
    pp.create_sgen(net, b2, p_mw=163., q_mvar=0., name='sgen2')
    pp.create_sgen(net, b3, p_mw=85., q_mvar=0., name='sgen3')
    
    return net

def create_data_source_high_load(n_timesteps):
    profiles = pd.DataFrame()
    profiles['load5_p'] = 100 + (0.05*np.random.random(n_timesteps)*90)
    profiles['load5_q'] = 35 + (0.05 * np.random.random(n_timesteps)*30)
    profiles['load7_p'] = 110 + (0.10*np.random.random(n_timesteps)*100)
    profiles['load7_q'] = 40 + (0.10 * np.random.random(n_timesteps)*35)
    profiles['load9_p'] = 130 + (0.05*np.random.random(n_timesteps)*125)
    profiles['load9_q'] = 55 + (0.05 * np.random.random(n_timesteps)*50)
    ds = DFData(profiles)
    return profiles, ds

def create_data_source_low_load(n_timesteps):
    profiles = pd.DataFrame()
    profiles['load5_p'] = 80 + (0.05*np.random.random(n_timesteps)*90)
    profiles['load5_q'] = 25 + (0.05 * np.random.random(n_timesteps)*30)
    profiles['load7_p'] = 95 + (0.10*np.random.random(n_timesteps)*100)
    profiles['load7_q'] = 30 + (0.10 * np.random.random(n_timesteps)*35)
    profiles['load9_p'] = 120 + (0.05*np.random.random(n_timesteps)*125)
    profiles['load9_q'] = 45 + (0.05 * np.random.random(n_timesteps)*50)
    ds = DFData(profiles)
    return profiles, ds

def create_data_source_gen_off(n_timesteps):
    profiles = pd.DataFrame()
    profiles['load5_p'] = 90 + (0.05*np.random.random(n_timesteps)*90)
    profiles['load5_q'] = 30 + (0.05 * np.random.random(n_timesteps)*30)
    profiles['load7_p'] = 100 + (0.05*np.random.random(n_timesteps)*100)
    profiles['load7_q'] = 35 + (0.05 * np.random.random(n_timesteps)*35)
    profiles['load9_p'] = 125 + (0.05*np.random.random(n_timesteps)*125)
    profiles['load9_q'] = 50 + (0.05 * np.random.random(n_timesteps)*50)
    ds = DFData(profiles)
    return profiles, ds

def create_data_source_line_disconnect(n_timesteps):
    profiles = pd.DataFrame()
    profiles['load5_p'] = 90 + (0.05*np.random.random(n_timesteps)*90)
    profiles['load5_q'] = 30 + (0.05 * np.random.random(n_timesteps)*30)
    profiles['load7_p'] = 100 + (0.05*np.random.random(n_timesteps)*100)
    profiles['load7_q'] = 35 + (0.05 * np.random.random(n_timesteps)*35)
    profiles['load9_p'] = 125 + (0.05*np.random.random(n_timesteps)*125)
    profiles['load9_q'] = 50 + (0.05 * np.random.random(n_timesteps)*50)
    ds = DFData(profiles)
    return profiles, ds


def create_controllers_load(net, ds):
    
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                          data_source=ds, profile_name=['load5_p'])
    ConstControl(net, element='load', variable='q_mvar', element_index=[0],
                          data_source=ds, profile_name=['load5_q'])
    ConstControl(net, element='load', variable='p_mw', element_index=[1],
                          data_source=ds, profile_name=['load7_p'])
    ConstControl(net, element='load', variable='q_mvar', element_index=[1],
                          data_source=ds, profile_name=['load7_q'])
    ConstControl(net, element='load', variable='p_mw', element_index=[2],
                          data_source=ds, profile_name=['load9_p'])
    ConstControl(net, element='load', variable='q_mvar', element_index=[2],
                          data_source=ds, profile_name=['load9_q'])
    
    
def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    
    return ow


def train_timeseries(output_dir):
    
    if output_dir=="./output_data_train_high_load/":
        net = CAPS_test_net()
        n_timesteps = 60
        profiles, ds = create_data_source_high_load(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
    if output_dir=="./output_data_train_low_load/":
        net = CAPS_test_net()
        n_timesteps = 60
        profiles, ds = create_data_source_low_load(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
    if output_dir=="./output_data_train_gen_off/":
        net = CAPS_test_net()
        n_timesteps = 60
        profiles, ds = create_data_source_gen_off(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        net.sgen.in_service[1] = False
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
    if output_dir=="./output_data_train_line_disconnect/":
        net = CAPS_test_net()
        n_timesteps = 60
        profiles, ds = create_data_source_line_disconnect(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        net.switch.closed[7] = False
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
        

   
 
    print(net.res_line.loading_percent)
    print(net.bus)
    print(net.sgen)
    print(net.load)
    print(net.line)
    print(net.ext_grid)



scenarios=[1,2,3,4]

for i in scenarios:
    if i==1:
        output_dir = "./output_data_train_high_load/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        train_timeseries(output_dir)
    
    if i==2:
        output_dir = "./output_data_train_low_load/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        train_timeseries(output_dir)
    
    if i==3:
        output_dir = "./output_data_train_gen_off/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        train_timeseries(output_dir)
        
    if i==4:
        output_dir = "./output_data_train_line_disconnect/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        train_timeseries(output_dir)

Voltage_file_list = []
for i in scenarios:
    if i==1:
        output_dir = "./output_data_train_high_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file1= pd.read_excel(vm_pu_file)
        Voltage_file_list.append(file1)
    if i==2:
        output_dir = "./output_data_train_low_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file2= pd.read_excel(vm_pu_file)
        Voltage_file_list.append(file2)
    if i==3:
        output_dir = "./output_data_train_gen_off/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file3= pd.read_excel(vm_pu_file)
        Voltage_file_list.append(file3)
    if i==4:
        output_dir = "./output_data_train_line_disconnect/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file4= pd.read_excel(vm_pu_file)
        Voltage_file_list.append(file4)

combined_data_vm_train = pd.concat(Voltage_file_list, ignore_index=True)
cols_to_convert = combined_data_vm_train.iloc[:, 1:].columns
arr1 = combined_data_vm_train[cols_to_convert].to_numpy()
combined_data_vm_train.to_excel('output_file_vm.xlsx', index=False)

Angle_file_list = []
for i in scenarios:
    if i==1:
        output_dir = "./output_data_train_high_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file1= pd.read_excel(vm_pu_file)
        file1['scenario']='1'
        Angle_file_list.append(file1)
    if i==2:
        output_dir = "./output_data_train_low_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file2= pd.read_excel(vm_pu_file)
        file2['scenario']='2'
        Angle_file_list.append(file2)
    if i==3:
        output_dir = "./output_data_train_gen_off/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file3= pd.read_excel(vm_pu_file)
        file3['scenario']='3'
        Angle_file_list.append(file3)
    if i==4:
        output_dir = "./output_data_train_line_disconnect/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file4= pd.read_excel(vm_pu_file)
        file4['scenario']='4'
        Angle_file_list.append(file4)

combined_data_va_train = pd.concat(Angle_file_list, ignore_index=True)
cols_to_convert = combined_data_va_train.iloc[:, 1:].columns
arr2 = combined_data_va_train[cols_to_convert].to_numpy()
voltage_training= np.concatenate((arr1, arr2), axis=1)
voltage_training.astype(float)
voltage_training_=np.transpose(voltage_training)
combined_data_va_train.to_excel('output_file_va.xlsx', index=False) 

file_vm_train=os.path.join("output_file_vm.xlsx")
file1_train= pd.read_excel(file_vm_train, index_col=0)
file_va_train=os.path.join("output_file_va.xlsx")
file2_train= pd.read_excel(file_va_train, index_col=0)
combined_data_train=pd.concat([file1_train, file2_train], axis=1, ignore_index=True)
combined_data_train.reset_index(inplace=True)
combined_data_train.to_excel('training_data.xlsx', index=False)
training_class = combined_data_train.iloc[:, -1]
training_class = training_class.astype(int)
print(len(training_class))

def test_timeseries(output_dir):
    
    if output_dir=="./output_data_test_high_load/":
        net = CAPS_test_net()
        n_timesteps = 15
        profiles, ds = create_data_source_high_load(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
    if output_dir=="./output_data_test_low_load/":
        net = CAPS_test_net()
        n_timesteps = 15
        profiles, ds = create_data_source_low_load(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
    if output_dir=="./output_data_test_gen_off/":
        net = CAPS_test_net()
        n_timesteps = 15
        profiles, ds = create_data_source_gen_off(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        net.sgen.in_service[1] = False
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
    if output_dir=="./output_data_test_line_disconnect/":
        net = CAPS_test_net()
        n_timesteps = 15
        profiles, ds = create_data_source_line_disconnect(n_timesteps)
        create_controllers_load(net, ds)
        time_steps = range(0, n_timesteps)
        net.switch.closed[7] = False
        create_output_writer(net, time_steps, output_dir)
        run_timeseries(net, time_steps)
        


scenarios=[1,2,3,4]

for i in scenarios:
    if i==1:
        output_dir = "./output_data_test_high_load/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        test_timeseries(output_dir)
    
    if i==2:
        output_dir = "./output_data_test_low_load/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        test_timeseries(output_dir)
    
    if i==3:
        output_dir = "./output_data_test_gen_off/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        test_timeseries(output_dir)
        
    if i==4:
        output_dir = "./output_data_test_line_disconnect/"
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        test_timeseries(output_dir)

Voltage_file_list_test = []
for i in scenarios:
    if i==1:
        output_dir = "./output_data_test_high_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file1= pd.read_excel(vm_pu_file)
        Voltage_file_list_test.append(file1)
    if i==2:
        output_dir = "./output_data_test_low_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file2= pd.read_excel(vm_pu_file)
        Voltage_file_list_test.append(file2)
    if i==3:
        output_dir = "./output_data_test_gen_off/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file3= pd.read_excel(vm_pu_file)
        Voltage_file_list_test.append(file3)
    if i==4:
        output_dir = "./output_data_test_line_disconnect/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
        file4= pd.read_excel(vm_pu_file)
        Voltage_file_list_test.append(file4)

combined_data_vm_test = pd.concat(Voltage_file_list_test, ignore_index=True)
combined_data_vm_test.to_excel('output_file_vm_test.xlsx', index=False)

Angle_file_list_test = []
for i in scenarios:
    if i==1:
        output_dir = "./output_data_test_high_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file1= pd.read_excel(vm_pu_file)
        file1['scenario']='1'
        Angle_file_list_test.append(file1)
    if i==2:
        output_dir = "./output_data_test_low_load/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file2= pd.read_excel(vm_pu_file)
        file2['scenario']='2'
        Angle_file_list_test.append(file2)
    if i==3:
        output_dir = "./output_data_test_gen_off/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file3= pd.read_excel(vm_pu_file)
        file3['scenario']='3'
        Angle_file_list_test.append(file3)
    if i==4:
        output_dir = "./output_data_test_line_disconnect/"
        vm_pu_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
        file4= pd.read_excel(vm_pu_file)
        file4['scenario']='4'
        Angle_file_list_test.append(file4)

combined_data_va_test = pd.concat(Angle_file_list_test, ignore_index=True)
combined_data_va_test.to_excel('output_file_va_test.xlsx', index=False) 


file_vm_test=os.path.join("output_file_vm_test.xlsx")
file1_test= pd.read_excel(file_vm_test, index_col=0)
file_va_test=os.path.join("output_file_va_test.xlsx")
file2_test= pd.read_excel(file_va_test, index_col=0)
combined_data_test=pd.concat([file1_test, file2_test], axis=1, ignore_index=True)
combined_data_test.reset_index(inplace=True)
combined_data_test.to_excel('testing_data.xlsx', index=False)
testing_class = combined_data_test.iloc[:, -1]
testing_class = testing_class.astype(int)


"""
-------------------------------------------------------------------------------
K-means clustering
"""
def eq_distance(x1, x2, inputs):
    dist = 0
    for i in range(inputs.shape[1]):
        dist = dist + (inputs[x1][i] - x2[i])**2
    distance = np.sqrt(dist)
    return distance

def mean(cluster, inputs):
    n = len(cluster)
    add = np.zeros(inputs.shape[1])
    if n == 0:
        mean = add
    else:
        for i in cluster:
            for j in range(inputs.shape[1]):
                add[j] = add[j] + inputs[i][j]
        mean = np.divide(add, n)
    return mean

def difference_(x1, x2):
    sum_squared_diff = 0
    for i in range(len(x1)):
        sum_squared_diff = sum_squared_diff + (x1[i] - x2[i]) ** 2
    difference = np.sqrt(sum_squared_diff)
    return difference

def kmeansclustering(inputs1):
    inputs = inputs1[:, :-1]
    n = inputs.shape[0]
    k = 4
    tole = 1e-4
    centroid = []
    cluster_all = []
    centroid_final = []
    while True:
        r = random.sample(range(0, n), k)
        for i in range(k):
            centroid.append(inputs[r[i], :])
        prev_centroid = centroid.copy()
        while True:
            cluster = [[] for _ in range(k)]
            for i in range(n):
                distance = []
                for j in range(k):
                    distance.append(eq_distance(i, centroid[j], inputs))
                index_clusters = np.argmin(distance)
                cluster[index_clusters].append(i)
            centroid = []
            for i in range(k):
                centroid.append(mean(cluster[i], inputs))
            if all(difference_(centroid[i], prev_centroid[i]) < tole for i in range(k)):
                break
            prev_centroid = centroid.copy()
        cluster_all.append(cluster)
        centroid_final.append(centroid)
        if len(cluster_all) > 1 and cluster_all[-1] == cluster_all[-2]:
            break
    return cluster_all[-1], centroid_final[-1]

# Perform the k means clustering using the datasets created by getVoltageProfile
kmeans_cluster, kmeans_centroid = kmeansclustering(voltage_training)

# Plot the generated voltage profile
y1 = []
x1 = []

for i in range(len(voltage_training_ [0])):
    
    y1 = voltage_training_[0:8, i]
    x1 = voltage_training_[9:17, i]
    
    if voltage_training_[-1, i] == 1:
        plt.scatter(x1, y1, s=5, c='blue')
        
    elif voltage_training_[-1, i] == 2:
        plt.scatter(x1, y1, s=5, c='black')
        
    elif voltage_training_[-1, i] == 3:
        plt.scatter(x1, y1, s=5, c='red')
        
    elif voltage_training_[-1, i] == 4:
        plt.scatter(x1, y1, s=5, c='green')    
    
plt.xlabel('Voltage Angle')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('voltage profile')
plt.show()

# plot the scatterplot
    
y = []
x = []


for j in range(len(kmeans_cluster [0])):
    y = voltage_training_[0:8, kmeans_cluster[0][j]]
    x = voltage_training_[9:17, kmeans_cluster[0][j]]
    plt.scatter(x, y, s=5, c='blue')
    
for j in range(len(kmeans_cluster[1])):
     y = voltage_training_[0:8, kmeans_cluster[1][j]]
     x = voltage_training_[9:17, kmeans_cluster[1][j]]
     plt.scatter(x, y, s=5, c='black')
    
for j in range(len(kmeans_cluster[2])):
     y = voltage_training_[0:8, kmeans_cluster[2][j]]
     x = voltage_training_[9:17, kmeans_cluster[2][j]]
     plt.scatter(x, y, s=5, c='red')
    
for j in range(len(kmeans_cluster[3])):
     y = voltage_training_[0:8, kmeans_cluster[3][j]]
     x = voltage_training_[9:17, kmeans_cluster[3][j]]
     plt.scatter(x, y, s=5, c='green')    
    
plt.xlabel('Voltage Angle')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('K-Means Clustering')
plt.show()

"""
------------------------------------------------------------------------------------
KNN-supervised learning
"""
k=35
nearest_id = []
nearest_distance = []
nearest_sorted = []
nearest_dist_k = []
KNN_class_found=[]
print(len(combined_data_train.index))

for x in range(len(combined_data_test.index)):
    nearest_id = []
    nearest_distance = []
    nearest_sorted = []
    nearest_dist_k = []
    for y in range(len(combined_data_train.index)):
        dist=0
        for z in range(18):            
            dist = dist + (combined_data_train.iloc[y,z]-combined_data_test.iloc[x,z])**2
        nearest_distance.append(math.sqrt(dist))
    nearest_sorted = sorted(nearest_distance)
    nearest_dist_k = nearest_sorted[0:k]
    for l in range(k):
        nearest_id.append(nearest_distance.index(nearest_dist_k[l]))
    data_type = [0]*5
    for i in nearest_id:
        data_type[training_class[i]] = data_type[training_class[i]] + 1
    
    data_type_max = data_type.index(max(data_type))
    KNN_class_found.append(data_type_max)
print(KNN_class_found)
right=0
wrong=0
for case in range(len(testing_class)):
    if KNN_class_found[case] == testing_class[case]:
        right = right + 1
    
    else:
        wrong = wrong + 1
        
accuracy = right/(right+wrong)
print("the KNN predicts with an accuracy of "+str(100*accuracy)+"%")

