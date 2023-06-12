# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:57:27 2023

@author: Amritha jayan and Yashwanth Damodaran
"""

import os
import numpy as np
import pandas as pd
import random
import tempfile

import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl

import matplotlib.pyplot as plt
random.seed(1)

# This function generates a training data for the given
# number of time steps and given state of operation
# x = 1: high load
# x = 2: low load
# x = 3: line disconnection
# x = 4: generator disconnection

#Generate the training data of the 4 operating states


def timeseries(x, no_time_steps, output_dir):
    # 1. create test net
    net = Test_network()

    # 2. create (random) data source
    profile, ds = create_data_source(x, net, no_time_steps)
    
    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(x, net, ds)

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, no_time_steps)
    
    # 4. the output writer with the desired results to be stored to files.
    ow= create_output_writer(net, time_steps, output_dir=output_dir)
    
    # 5. the main time series and test data function
    pp.set_user_pf_options(net, calculate_voltage_angles=True)
    run_timeseries(net, time_steps)  


def Test_network():
    
    net = pp.create_empty_network()

    b0 = pp.create_bus(net, 110)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    b4 = pp.create_bus(net, 110)
    b5 = pp.create_bus(net, 110)  
    b6 = pp.create_bus(net, 110)
    b7 = pp.create_bus(net, 110)
    b8 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b0)
    pp.create_line(net, b0, b3, 10, '149-AL1/24-ST1A 110.0', name = 'line1')
    pp.create_line(net, b3, b4, 10, '149-AL1/24-ST1A 110.0', name = 'line2')
    pp.create_line(net, b3, b8, 10, '149-AL1/24-ST1A 110.0', name = 'line3')
    pp.create_line(net, b4, b5, 10, '149-AL1/24-ST1A 110.0', name = 'line4')
    pp.create_line(net, b5, b2, 10, '149-AL1/24-ST1A 110.0', name = 'line5')
    pp.create_line(net, b7, b8, 10, '149-AL1/24-ST1A 110.0', name = 'line6')
    pp.create_line(net, b5, b6, 10, '149-AL1/24-ST1A 110.0', name = 'line7')
    pp.create_line(net, b6, b7, 10, '149-AL1/24-ST1A 110.0', name = 'line8')
    pp.create_line(net, b7, b1, 10, '149-AL1/24-ST1A 110.0', name = 'line9')
    
    pp.create_load(net, b4, p_mw=90., q_mvar=30., name='load1')
    pp.create_load(net, b6, p_mw=100., q_mvar=35., name='load2')
    pp.create_load(net, b8, p_mw=125., q_mvar=50., name='load3')
   
    pp.create_sgen(net, b1, p_mw=163, q_mvar=0, name='generator1')
    pp.create_sgen(net, b2, p_mw=85, q_mvar=0, name='generator2')

    print (net)
    return net

def create_data_source(x, net, no_time_steps):
    
    profile = pd.DataFrame()
    
    if x == 1:
    
        # high load
        profile['load1_p'] = np.random.normal(99., 9, no_time_steps)
        profile['load2_p'] = np.random.normal(110., 10, no_time_steps)
        profile['load3_p'] = np.random.normal(137.5, 12.5, no_time_steps)
        profile['load1_q'] = np.random.normal(33., 3, no_time_steps)
        profile['load2_q'] = np.random.normal(38.5, 3.5, no_time_steps)
        profile['load3_q'] = np.random.normal(55., 5, no_time_steps)
        
    elif x == 2:
        # low load state
        profile['load1_p'] = np.random.normal(81., 9, no_time_steps)
        profile['load2_p'] = np.random.normal(90., 10, no_time_steps)
        profile['load3_p'] = np.random.normal(112.5, 12.5, no_time_steps)
        profile['load1_q'] = np.random.normal(27., 3, no_time_steps)
        profile['load2_q'] = np.random.normal(31.5, 3.5, no_time_steps)
        profile['load3_q'] = np.random.normal(45., 5, no_time_steps)   
        
    elif x == 3:
        # Line Disconnection
        profile['load1_p'] = np.random.normal(90., 4.5, no_time_steps)
        profile['load2_p'] = np.random.normal(100., 5, no_time_steps)
        profile['load3_p'] = np.random.normal(125., 6.25, no_time_steps)
        profile['load1_q'] = np.random.normal(30., 1.5, no_time_steps)
        profile['load2_q'] = np.random.normal(35, 1.75, no_time_steps)
        profile['load3_q'] = np.random.normal(50., 2.5, no_time_steps)       
        
        pp.create_switch(net, bus=5, closed=False, element=6, et='l', type='CB')
    
    elif x == 4:
        
        # Generator Disconnection
        profile['load1_p'] = np.random.normal(90, 4.5, no_time_steps)
        profile['load2_p'] = np.random.normal(100, 5, no_time_steps)
        profile['load3_p'] = np.random.normal(125, 6.25, no_time_steps)
        profile['load1_q'] = np.random.normal(30., 1.5, no_time_steps)
        profile['load2_q'] = np.random.normal(35, 1.75, no_time_steps)
        profile['load3_q'] = np.random.normal(50., 2.5, no_time_steps) 
        profile['generator2_p'] = [0]*no_time_steps

    ds = DFData(profile)
    
    return profile, ds
        
def create_controllers(x,net, ds):
    
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                     data_source=ds, profile_name=["load1_p"])
    ConstControl(net, element='load', variable='p_mw', element_index=[1],
                     data_source=ds, profile_name=["load2_p"])
    ConstControl(net, element='load', variable='p_mw', element_index=[2],
                     data_source=ds, profile_name=["load3_p"])
    ConstControl(net, element="load", variable="q_mvar", element_index=[0],
                 data_source=ds, profile_name=["load1_q"])
    ConstControl(net, element="load", variable="q_mvar", element_index=[1], 
                 data_source=ds, profile_name=["load2_q"])
    ConstControl(net, element="load", variable="q_mvar", element_index=[2], 
                 data_source=ds, profile_name=["load3_q"])

    
    if x == 4:
        # Generator Disconnection
        ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                         data_source=ds, profile_name=["generator2_p"])

def create_output_writer(net, time_steps, output_dir):

    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    
    return ow

def get_voltage_data(x, no_time_steps):       

    output_dir = "./time_series/"
    print("Results can be found in your local temp folder: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    timeseries(x, no_time_steps, output_dir)
    
    # 7.
    # Voltage magnitude
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    vm_pu_tot = pd.read_excel(vm_pu_file)
    vm_pu = vm_pu_tot[[0,1,2,3,4,5,6,7,8]]
    vm_pu.plot(label="vm_pu")
    plt.xlabel("time step")
    plt.ylabel("voltage mag. [p.u.]")
    plt.title("Voltage Magnitude")
    plt.grid()
    plt.show()
        
    # Voltage angle
    va_degree_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
    va_degree_tot = pd.read_excel(va_degree_file)
    va_degree = va_degree_tot[[0,1,2,3,4,5,6,7,8]]
    va_degree.plot(label="va_degree")
    plt.xlabel("time step")
    plt.ylabel("voltage angle. [degree]")
    plt.title("Voltage Angle")
    plt.grid()
    plt.show()
        
    # generate training data
    # Get the voltage magnitude and angle from the data frame
    voltage_mag = []
    voltage_deg = []

    for i in range(9):
        temp_mag = []
        temp_deg = []
        for j in range(no_time_steps):
            temp_mag.append(vm_pu_tot[i][j])
            temp_deg.append(va_degree_tot[i][j])
        
        voltage_mag.append(temp_mag)
        voltage_deg.append(temp_deg)

    voltage_profile_temp = voltage_mag + voltage_deg
    voltage_profile = np.empty(shape=(19,no_time_steps))
    for i in range(18):
        for j in range(no_time_steps):
            voltage_profile[i][j] = voltage_profile_temp[i][j]

    for i in range(no_time_steps):
        voltage_profile[-1][i] = x
        
    return voltage_profile
#
#Voltage training data for all the four scenarios
v1 = get_voltage_data(1, 40)  
v2 = get_voltage_data(2, 40)
v3 = get_voltage_data(3, 40)
v4 = get_voltage_data(4, 40)

# Combine the operating states into a single training or testing dataset
voltage_training = np.concatenate((v1, v2, v3, v4), axis=1)
label_training = voltage_training[-1][:]
label_training = label_training.astype(int)

# Plot the generated voltage profile

for i in range(len(voltage_training[0])):
    
    y = []
    x = []
    
    y = voltage_training[0:8, i]
    x = voltage_training[9:17, i]
    
    if voltage_training[-1, i] == 1:
        plt.scatter(x, y, s=5, c='blue')
        
    elif voltage_training[-1, i] == 2:
        plt.scatter(x, y, s=5, c='black')
        
    elif voltage_training[-1, i] == 3:
        plt.scatter(x, y, s=5, c='red')
        
    elif voltage_training[-1, i] == 4:
        plt.scatter(x, y, s=5, c='green')    
    
plt.xlabel('Voltage Angle')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.show()

# Kmean_clustering

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
voltage_data = np.transpose(voltage_training)
kmeans_cluster, kmeans_centroid = kmeansclustering(voltage_data)

# plot the scatterplot
    
y = []
x = []


for j in range(len(kmeans_cluster [0])):
    y = voltage_training[0:8, kmeans_cluster[0][j]]
    x = voltage_training[9:17, kmeans_cluster[0][j]]
    plt.scatter(x, y, s=5, c='blue')
    
for j in range(len(kmeans_cluster[1])):
     y = voltage_training[0:8, kmeans_cluster[1][j]]
     x = voltage_training[9:17, kmeans_cluster[1][j]]
     plt.scatter(x, y, s=5, c='black')
    
for j in range(len(kmeans_cluster[2])):
     y = voltage_training[0:8, kmeans_cluster[2][j]]
     x = voltage_training[9:17, kmeans_cluster[2][j]]
     plt.scatter(x, y, s=5, c='red')
    
for j in range(len(kmeans_cluster[3])):
     y = voltage_training[0:8, kmeans_cluster[3][j]]
     x = voltage_training[9:17, kmeans_cluster[3][j]]
     plt.scatter(x, y, s=5, c='green')    
    
plt.xlabel('Voltage Angle')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

