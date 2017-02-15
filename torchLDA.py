#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation on PyTorch using Gibbs Motion
# This code is available under the MIT License.
# (c)2017 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University


import torch.nn as nn
from torch.autograd import Variable
from torch import mm,randn,from_numpy,mv,t
import torch
import numpy as np
from random import randint,seed
from aux import join2,addrandomcol,normalize_rows
import time



#--tonumpy----------------------------------------------------------
"""From 2dpyTorch tensor to numpy 2darray
Input: 2darray
Output: 2darray """
#-------------------------------------------------------------------
def tonumpy(m):
	matrix=[]
	for row in m:
		rows=[]
		for col in row:
			rows.append(col)
		matrix.append(rows)
	return np.array(matrix)

#--toDistribution-------------------------------------------------------
"""One hot encoding of value called position in a vector sized K
Input: int,int
Output: 1darray"""
#-----------------------------------------------------------------------
def toDistribution(position,K):
	distribution=np.zeros(K)
	distribution[position]=1
	return np.array(distribution)

#--LDA CLASS-------------------------------------------------------
"""Implements LDA as a pyTorch network
"""
#-----------------------------------------------------------------------
class LDA(nn.Module):
	def __init__(self,alpha,beta):
		super(LDA,self).__init__()
		
		self.alpha=alpha
		self.beta=beta
		
	def forward(self,dz,wz,z,pz,d,w,i):
	
		dz_d=mm(d,dz)  #indices = torch.LongTensor([d]) // torch.index_select(dz, 0, indices)
		wz_w=mm(w,wz)  #indices = torch.LongTensor([w]) // torch.index_select(wz, 0, indices)
		pz_i=mm(i,pz)  #indices = torch.LongTensor([i]) // torch.index_select(pz, 0, indices)
				
		#**1. Unbias estimation
		dz_d_before=dz_d-pz_i
		wz_w_before=wz_w-pz_i
		z_before=z-pz_i

		#**2. Compute Target
		pz_i_after=((dz_d_before+self.alpha)*(wz_w_before+self.beta))/(z_before+self.beta*6)
		pz_i_after=pz_i_after/torch.sum(pz_i_after)

		#**3.  Move towards target
		dz_d_after=dz_d_before+pz_i_after
		wz_w_after=wz_w_before+pz_i_after
		z_after=z_before+pz_i_after
		
		dz=dz+mm(t(d),dz_d_after)-mm(t(d),dz_d)
	
		wz=wz+mm(t(w),wz_w_after)-mm(t(w),wz_w)
		pz=pz+mm(t(i),pz_i_after)-mm(t(i),pz_i)
		
		
		return dz,wz,pz,z_after
		


#--torchLDA-------------------------------------------------------------
"""Runs LDA over the given data
data: 2darray shape (n,3) where n is the number of records. each record is a tuple (Document,word,topic)
K: number of topics
it: number of iterations
alpha,beta: Hyperparameters
dict_: True if data must be passed through dictionate()
verbose: True if it will output the extracted topics - not implemented yet
randomness= 1 if we want to intialize the topics, 0 if we dont want to
PATH: Path to save results
"""
#-------------------------------------------------------------------
def torchLDA(data,K,it,alpha,beta,dict_=True,verbose=True,randomness=1,PATH=""):

	#** 2. Random topics and dictionate
	data=np.asarray(data)
	if randomness>0:
		data=addrandomcol(data,K,-1,randomness)#K

	dict_=False
	if dict_:
		data,idx2vals,vals2idx,_=dictionate(data)
	else:
		idx2vals=None
		vals2idx=None
	
	data=data.astype(np.int)

	data=data.copy()
	d_z=join2(data[:][:,[0,2]])
	w_z=join2(data[:][:,[1,2]])
	z=join2(data[:][:,[2]])
	
	
	datamotion=map(lambda row: (row[0],row[1],toDistribution(row[2],K)),data)
	pz=np.array([row[2] for row in datamotion])

	D,W,I=len(d_z),len(w_z),len(data)
	i_hot,w_hot,d_hot=np.eye(I),np.eye(W),np.eye(D)
	
	dz=from_numpy(d_z)
	wz=from_numpy(w_z)
	z=from_numpy(z)
	pz=from_numpy(pz)
	
	
	if PATH!="":
		np.save(PATH+"w_z_ldatorch_"+str(K)+"_0",w_z)
		np.save(PATH+"z_d_ldatorch_"+str(K)+"_0",d_z.T)
	
	lda=LDA(alpha,beta)
	
	for j in range(it):
		start=time.time()
		for idx,row in enumerate(datamotion):
		
			curr_d,curr_w,_=row	
			d=from_numpy(d_hot[None,curr_d])
			w=from_numpy(w_hot[None,curr_w])
			i=from_numpy(i_hot[None,idx])
		
			dz,wz,pz,z=lda.forward(dz,wz,z,pz,d,w,i)
			
		
		if PATH!="":
			w_z=tonumpy(wz).T
			z_d=tonumpy(dz)
			np.save(PATH+"w_z_ldatorch_"+str(K)+"_"+str(idx+1),w_z)
			np.save(PATH+"z_d_ldatorch_"+str(K)+"_"+str(idx+1),z_d)
		print "Iteration",j,"took",time.time()-start
	return data,wz,dz,idx2vals,vals2idx

if __name__=="__main__":

	data=np.array([[0,0,0],
				[0,1,0],
				[0,1,0],
				[0,1,0],
				[0,1,0],
				[0,1,0],
				[0,1,0],
				[1,0,0],
				[1,0,0],
				[1,0,0],
				[1,1,0],
				[1,1,0],
				[1,1,0],
				[1,1,0],
				[1,2,0],
				[1,2,0],
				[1,4,0],
				[2,2,0],
				[2,1,0],
				[2,1,0],
				[2,2,0],
				[2,3,0],
				[2,4,0],							
				[3,2,0],
				[3,2,0],
				[3,2,0],
				[3,3,0],
				[3,3,0],					
				[3,3,0],
				[3,3,0],		
				[3,4,0],
				[3,5,0],
				[4,3,0],
				[4,3,0],
				[4,3,0],
				[4,3,0],
				[4,3,0],
				[4,3,0],
				[4,3,0],
				[4,3,0],
				[4,4,0],
				[4,4,0]])
			
	
	print "\nGibbs Motion on PyTorch...\n"	
			
	print "Input"
	print data		
	K,it,alpha,beta=(2,10,0.5,0.5)
	data_after_lda,w_z,z_d,idx2vals,vals2idx=torchLDA(data,K,it,alpha,beta,randomness=1)
	print "Output"
	print data_after_lda
