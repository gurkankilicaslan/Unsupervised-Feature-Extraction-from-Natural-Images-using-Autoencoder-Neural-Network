#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  18 11:53:37 2023

@author: gurkankilicaslan
"""

import h5py
import random
import numpy as np
from matplotlib.pyplot import imshow, show, subplot, figure, axis, plot, xlabel,ylabel,title,savefig
import matplotlib.pyplot as plt
 
def AeNN():
    
    
    #read data from .h5 file
    filename = "data.h5"
    
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
    
        # Get the data
        #data = list(f[a_group_key])
        data = np.array(f[a_group_key])
        
    def normalize (data):
        #turn rgb to graysclae
        data_gray = np.array(data)[:,0,:,:]*0.2126 + np.array(data)[:,1,:,:]*0.7152 + np.array(data)[:,2,:,:]*0.0722
        mean = np.mean(data_gray,axis=(1,2)) #find mean
        
        for i in range (len(mean)):
            data_gray[i,:,:] -= mean[i] #remove the mean pixel intensity of each image from itself
            
        std = np.std(data_gray)  # find std
        data_gray = np.clip(data_gray, - 3 * std, 3 * std)  # clip -+3 std
        #normalize and map to 0.1 - 0.9
        normalized = data_gray*(0.90 - 0.10)/(2*np.max(data_gray))
        normalized += (0.10 - np.min(normalized))
        return normalized
    
    def random_index():
        random_index = []
        for i in range(200):
            index =random.randint(0,10239)
            random_index.append(index)
        return random_index
        
    
    def show_images(data,random_pics):
        figure(figsize=(20,10))
        j=0
        for i in random_pics:
            j+=1
            img = (data[i]).T
            subplot(10, 20, j)
            imshow(img)
            axis("off")
        savefig("fig1")
        
    def show_gray_images(data,random_pics):
        figure(figsize=(20,10))
        j=0
        for i in random_pics:
            j+=1
            img = (data[i]).T
            subplot(10, 20, j)
            imshow(img,cmap="gray")
            axis("off")
        savefig("fig2")
        
    ################################### PART B ####################################
        
    def w0 (Lpre,Lpost):
        w0 = np.sqrt(6/(Lpre + Lpost))
        return w0
    
    def initialize_weights(Lin, Lhid):
        
        Lout = Lin
        np.random.seed(42) 
        
        W1 = np.random.uniform(-w0(Lin,Lhid),w0(Lin,Lhid), size = (Lin,Lhid))
        W2 = np.random.uniform(-w0(Lhid,Lout),w0(Lhid,Lout), size = (Lhid,Lout))
        b1 = np.random.uniform(-w0(1,Lhid),w0(1,Lhid), size = (1,Lhid))
        b2 = np.random.uniform(-w0(1,Lout),w0(1,Lout), size = (1,Lout))
        
        we = [W1,W2,b1,b2]
        
        return we
    
    def initialize_params(Lin,Lhid,lmbd, beta, rho):    
        params = [Lin,Lhid,lmbd,beta,rho]
        
        return params
    
    def sigmoid(x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    def sigmoid_backward(x):
        d_sig = x*(1-x)
        return d_sig
    
    def forward(data, we):
        
        w1,w2,b1,b2 = we
    
        z1 = np.dot(data,w1) + b1 #first layer linear forward
        A1 = sigmoid(z1) #first layer activation
        z2 =  np.dot(A1,w2) + b2 #output linear forward
        output = sigmoid(z2) #output layer activation
    
        return A1, output
    
    def aeCost(we, data, params):
        
        Lin,Lhid,lmbd,beta,rho = params
        w1,w2,b1,b2 = we
        N = len(data)
        
        A1, output = forward(data, we)
        mean = np.mean(A1, axis=0)
        
        #calculate cost 
        average_squared_error = (1/(2*N))*np.sum((data-output)**2)
        tykhonov = (lmbd/2)*(np.sum(w1**2) + np.sum(w2**2))
        kl_divergence = beta*np.sum((rho*np.log(mean/rho))+((1-rho)*np.log((1-mean)/(1-rho))))
        J = average_squared_error + tykhonov + kl_divergence
        d_hid = (np.dot(w2,(-(data-output)*sigmoid_backward(output)).T)+ (np.tile(beta*(-(rho/mean.T)+((1-rho)/(1-mean.T))), (10240,1)).T)) * sigmoid_backward(A1).T   
        
        d_w1 = (1/N)*(np.dot(data.T,d_hid.T) + lmbd*w1)
        d_w2 = (1/N)*(np.dot((-(data-output)*sigmoid_backward(output)).T,A1).T + lmbd*w2)
        d_b1 = np.mean(d_hid, axis=1)
        d_b2 = np.mean((-(data-output)*sigmoid_backward(output)), axis=0)
        
        Jgrad = [d_w1,d_w2,d_b1,d_b2]
        
        return J, Jgrad
    
    
    def backward(data, lr_rate, we, params):
        #get gradients
        J, Jgrad = aeCost(we, data, params)
        #update weights 
        we[0] -= lr_rate*Jgrad[0]
        we[1] -= lr_rate*Jgrad[1]
        we[2] -= lr_rate*Jgrad[2]
        we[3] -= lr_rate*Jgrad[3]
        return J, we
    
    def train(data_gray,epoch,Lin,Lhid,lmbd, beta, rho,lr_rate):
        losses = []
        epochs = []
        data_flat = np.reshape(data_gray, (data_gray.shape[0],data_gray.shape[1]**2))
        we = initialize_weights(Lin,Lhid)
        params = initialize_params(Lin,Lhid,lmbd, beta, rho)
    
    
        for i in range (epoch):
            J, we = backward(data_flat, lr_rate, we, params)
            epochs.append(i)
            losses.append(J)
            print("Epoch: {} --------------> Loss: {} ".format(i+1,J))
    
        return we,losses,epochs
    
    def plot_weights(we,name):
        w1,w2,b1,b2 = we
        figure(figsize=(18, 16))
        plot_shape = int(np.sqrt(w1.shape[1]))
        for i in range(w1.shape[1]):
            subplot(plot_shape,plot_shape,i+1)
            imshow(np.reshape(w1[:,i],(16,16)), cmap='gray')
            axis('off')
        savefig(name)

    index = random_index()
    show_images(data,index)
    data_gray = normalize (data)
    show_gray_images(data_gray,index)

    we,losses,epochs = train(data_gray,2000,256,64,5e-4, 0.01, 0.03,0.7)
    plot_weights(we,"fig3")
    plt.figure()
    plt.plot(epochs, losses)
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('ilkloss.png')
    plt.show()
    
    
    ############ Lhid = 16, alpha = 0 ############################    
    we,losses,epochs = train(data_gray,2000,256,16,0, 0.01, 0.03,0.7)
    plot_weights(we,"fig4")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 16, \lambda = 0$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('16_0.png')
    plt.show()
    
    ############ Lhid = 49, alpha = 0 ############################    
    we,losses,epochs = train(data_gray,2000,256,49,0, 0.01, 0.03,0.7)
    plot_weights(we,"fig5")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 49, \lambda = 0$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('49_0.png')
    plt.show()
    
    ############ Lhid = 81, alpha = 0 ############################    
    we,losses,epochs = train(data_gray,2000,256,81,0, 0.01, 0.03,0.7)
    plot_weights(we,"fig6")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 81, \lambda = 0$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('81_0.png')
    plt.show()
    
    ############ Lhid = 16, alpha = 1e-3 ############################    
    we,losses,epochs = train(data_gray,2000,256,16,1e-3, 0.01, 0.03,0.7)
    plot_weights(we,"fig7")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 16, \lambda = 1e-3$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('16_1e3.png')
    plt.show()
    
    ############ Lhid = 49, alpha = 1e-3 ############################    
    we,losses,epochs = train(data_gray,2000,256,49,1e-3, 0.01, 0.03,0.7)
    plot_weights(we,"fig8")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 49, \lambda = 1e-3$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('49_1e3.png')
    plt.show()
    
    ############ Lhid = 81, alpha = 1e-3 ############################    
    we,losses,epochs = train(data_gray,2000,256,81,1e-3, 0.01, 0.03,0.7)
    plot_weights(we,"fig9")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 81, \lambda = 1e-3$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('81_1e3.png')
    plt.show()
    
    ############ Lhid = 16, alpha = 1e-6 ############################    
    we,losses,epochs = train(data_gray,2000,256,16,1e-6, 0.01, 0.03,0.7)
    plot_weights(we,"fig10")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 16, \lambda = 1e-6$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('16_1e6.png')
    plt.show()
    
    ############ Lhid = 49, alpha = 1e-6 ############################    
    we,losses,epochs = train(data_gray,2000,256,49,1e-6, 0.01, 0.03,0.7)
    plot_weights(we,"fig11")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 49, \lambda = 1e-6$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('49_1e6.png')
    plt.show()
    
    ############ Lhid = 81, alpha = 1e-6 ############################    
    we,losses,epochs = train(data_gray,2000,256,81,1e-6, 0.01, 0.03,0.7)
    plot_weights(we,"fig12")
    plt.figure()
    plt.plot(epochs, losses)
    latex_text = r'$L_{hid} = 81, \lambda = 1e-6$'
    plt.legend([latex_text])
    xlabel("epoch")
    ylabel("loss")
    plt.savefig('81_1e6.png')
    plt.show()
   
AeNN()
