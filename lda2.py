#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation using Collapsed Gibbs Sampling and Gibbs Motion
# This code is available under the MIT License.
# (c)2017 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University

import numpy as np
from aux import dictionate,join2,addrandomcol,normalize_rows
np.set_printoptions(precision=3)
from deco import *
import time

__author__ = "Daniel Rugeles"
__copyright__ = "Copyright 2017, Latent Dirichlet Allocation"
__credits__ = ["Daniel Rugeles"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Daniel Rugeles"
__email__ = "daniel007@e.ntu.edu.sg"
__status__ = "Released"


#--computePosterior-------------------------------------------------
"""Computes the posterior p(z_i|z,alpha,beta)
Input: 2darray,2darray,1darray,float,float,int
Output: 1darray"""
#-------------------------------------------------------------------
def computePosterior(z_d,w_z,z_,row,alpha,beta,rowid):
	d,w,z=row
	p_z=np.multiply(z_d[d]+alpha,(w_z[:][:,w]+beta)/(z_+beta*len(w_z.T)))
	p_z=p_z/p_z.sum()
	
	return p_z
	
#--sampling---------------------------------------------------------
"""Performs Collapsed Gibbs Sampling
Input: 2darray,2darray,2darray,1darray,float,float
Output: 1darray"""
#-------------------------------------------------------------------
def sampling(data,z_d,w_z,z_,alpha,beta):
	for rowid,row in enumerate(data):
		d,w,z=row
		z_d[d][z]-=1
		w_z[z][w]-=1
		z_[z]-=1
		p_z=computePosterior(z_d,w_z,z_,row,alpha,beta,rowid)
		newk=np.random.multinomial(1, p_z).argmax()
		z_d[d][newk]+=1
		w_z[newk][w]+=1
		z_[z]+=1
		
		data[rowid][2]=newk
	return data,z_d,w_z

def toDistribution(position,K):
	distribution=np.zeros(K)
	distribution[position]=1
	return np.array(distribution)
	
#--motion---------------------------------------------------------
"""Performs Gibbs Motion
Input: 2darray,2darray,2darray,1darray,float,float
Output: 1darray"""
#-------------------------------------------------------------------
def motion(data,z_d,w_z,z_,alpha,beta):
	for rowid,row in enumerate(data):
		d,w,p_z=row
		z_d[d]-=p_z
		w_z[:][:,w]-=p_z
		z_-=p_z
		#assert(np.sum(z_d<0)==0)
		#assert(np.sum(w_z<0)==0)
		p_z =computePosterior(z_d,w_z,z_,row,alpha,beta,rowid)
		z_d[d]+=p_z
		w_z[:][:,w]+=p_z
		z_+=p_z
		
		data[rowid][2]=p_z
	return data,z_d,w_z


#--lda--------------------------------------------------------------
"""Runs LDA over the given data
data: 2darray shape (n,3) where n is the number of records. each record is a tuple (Document,word,topic)
K: number of topics
it: number of iterations
alpha,beta: hyperparameters
dict_: True if data must be passed through dictionate()
verbose: True if it will output the extracted topics - not implemented yet
randomness= 1 if we want to intialize the topics, 0 if we dont want to
PATH: Path to save results
algo: 'cgs' for collapsed gibbs sampling, 'motion' for Gibbs Motion
"""
#-------------------------------------------------------------------
@cache
def lda(data,K,it,alpha,beta,dict_=True,verbose=True,randomness=1,PATH="",algo='cgs'):

	#** 1. Define Internal Parameters
	alpha=0.5
	beta=alpha

	#** 2. Random topics and dictionate
	data=np.asarray(data)
	if randomness>0:
		data=addrandomcol(data,K,-1,randomness)#K

	if dict_:
		data,idx2vals,vals2idx,_=dictionate(data)
	else:
		idx2vals=None
		vals2idx=None
	
	data=data.astype(float)		
	data=data.astype(np.int)
	
	z_d=join2(data[:][:,[0,2]])
	w_z=join2(data[:][:,[2,1]])
	z_=join2(data[:][:,[2]])

	if algo=="motion":
		data=map(lambda row: [row[0],row[1],toDistribution(row[2],K)],data)

	#** 3. Inference
	if PATH!="":
		np.save(PATH+"w_z_lda"+algo+"_"+str(K)+"_0",w_z)
		np.save(PATH+"z_d_lda"+algo+"_"+str(K)+"_0",z_d)
	for i in range(it):
		start=time.time()
		if algo=="cgs":
			data,z_d,w_z=sampling(data,z_d,w_z,z_,alpha,beta)
		elif algo=="motion":
			data,z_d,w_z=motion(data,z_d,w_z,z_,alpha,beta)
		else:
			print "Only cgs and motion are implemented "
			assert(False)
		
		if PATH!="":
			np.save(PATH+"w_z_lda"+algo+"_"+str(K)+"_"+str(i+1),w_z)
			np.save(PATH+"z_d_lda"+algo+"_"+str(K)+"_"+str(i+1),z_d)
		print "Iteration",i,"took",time.time()-start
				

	return data,w_z,z_d,idx2vals,vals2idx
	

"""----------------------------*
*                              *
*   |\  /|   /\    |  |\  |    * 
*   | \/ |  /__\   |  | \ |    *
*   |    | /    \  |  |  \|    *
*                              *
*----------------------------"""
if __name__=="__main__":
	

	K,it,alpha,beta=(2,10,0.5,0.5)

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
	
	
	print "\nGibbs Motion ...\n"	
			
	print "Input"
	print data			
	data_after_lda,w_z,z_d,idx2vals,vals2idx=lda(data,K,it,alpha,beta,randomness=1,algo="cgs")
	print "Output"
	print data_after_lda
	
	
	print "\nGibbs Sampling ...\n"	
	print "Input"
	print data		
	data_after_lda,w_z,z_d,idx2vals,vals2idx=lda(data,K,it,alpha,beta,randomness=1,algo="cgs")
	print "Output"
	print data_after_lda
		

	
