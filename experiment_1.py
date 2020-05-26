# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:19:22 2019

@author: Haruka Murayama
"""

"""
Code for Experiment 1
Comparing proposal and random forest
Generating parameters and data from the priors
"""


import numpy as np
import scipy.stats as ss
import scipy.special as sss
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#Constants
p=3
q=5
K_max=5
data_max=100
Q=1
P=10
T=30
D=1
M=100
start=10
interval=10
R=2.
STD=1e180


#Generating true cluster number
for K_iter in range(Q):
    K=np.random.randint(K_max)+1
    

    
   #Initial values of hyper parameters
    alpha_true=np.random.random()*10+0.05
    m_true=np.random.random((K,p))
    beta_true=np.random.random()+0.001
    A_true=np.eye(p)
    nu_true=p+np.random.random()
    mu_v_true=np.zeros(q)
    Sigma_v_true=np.eye(q)
    mu_w_true=np.zeros((K,q+1))
    Lambda_w_true=np.eye(q+1)
        
        
    
    #Writing hyper parameters to files
    f_hyper_param=open("hyper_param.txt","a")
    f_hyper_param.write("alpha_true="+str(alpha_true)+"\n")
    f_hyper_param.write("m_true="+str(m_true)+"\n")
    f_hyper_param.write("beta_true="+str(beta_true)+"\n")
    f_hyper_param.write("nu_true="+str(nu_true)+"\n")
    f_hyper_param.write("K="+str(K)+"\n")
    f_hyper_param.close()


    
    for param_iter in range(P):
        print('//')
        print('param_iter=',param_iter)
        
        #Generating parameters from priors
        Lambda_true=np.zeros((K,p,p))
        mu_true=np.zeros((K,p))
        w_true=np.zeros((K,q+1))
        pi_true=np.random.dirichlet(alpha_true*np.ones(K))
        for k in range(K):
           Lambda_true[k]=ss.wishart.rvs(df=nu_true, scale=A_true)
           mu_true[k]=np.random.multivariate_normal(m_true[k],np.linalg.inv(beta_true * Lambda_true[k]))
           w_true[k]=np.random.multivariate_normal(mu_w_true[k],np.linalg.inv(Lambda_w_true))
        sigma_true=0.5
        
        for data_iter in range(D):
            print("+++++")
            print("data_iter=",data_iter)

            #Generating data
            u_data=np.zeros((data_max,p))
            v_data=np.zeros((data_max,q))
            v_tilde_data=np.zeros((data_max,q+1))
            z_true_data=np.zeros((data_max,K))
            Z_true_data=np.zeros(data_max)
            y_data=np.zeros(data_max)
            u_new=np.zeros((M,p))
            v_new=np.zeros((M,q))
            v_tilde_new=np.zeros((M,q+1))
            z_true_new=np.zeros((M,K))
            Z_true_new=np.zeros(M)
            y_new=np.zeros(M)
            
            #Train data
            for c in range(data_max):
                Z_true_data[c]=np.random.choice(list(range(K)),1,p=pi_true)
                for k in range(K):
                    if (Z_true_data[c]==k):
                        z_true_data[c,k]=1
                        u_data[c]=np.random.multivariate_normal(mu_true[k],np.linalg.inv(Lambda_true[k]),check_valid='warn')
                        v_data[c]=np.random.multivariate_normal(mu_v_true,Sigma_v_true)
                        v_tilde_data[c]=np.insert(v_data[c],0,[1])
                        y_data[c]=np.random.normal(np.dot(w_true[k].T,v_tilde_data[c]),np.square(sigma_true))
                    else:
                        z_true_data[c,k]=0
            #New data
            for j in range(M):
                Z_true_new[j]=np.random.choice(list(range(K)),1,p=pi_true)
                for k in range(K):
                    if (Z_true_new[j]==k):
                        z_true_new[j,k]=1
                        u_new[j]=np.random.multivariate_normal(mu_true[k],np.linalg.inv(Lambda_true[k]),check_valid='warn')
                        v_new[j]=np.random.multivariate_normal(mu_v_true,Sigma_v_true)
                        v_tilde_new[j]=np.insert(v_new[j],0,[1])
                        y_new[j]=np.random.normal(np.dot(w_true[k].T,v_tilde_new[j]),np.square(sigma_true))
                    else:
                        z_true_new[j,k]=0 
            
            #Transform to DataFrame
            u_new_pd=pd.DataFrame(u_new)
            v_new_pd=pd.DataFrame(v_new)
            y_new_pd=pd.DataFrame(y_new)
            X_new_pd=pd.concat([u_new_pd,v_new_pd],axis=1)
    
            error_vb_uniform=np.zeros(data_max)
            error_vb_K_tilde=np.zeros((data_max,K_max+1))
            K_prob_uniform=np.zeros((data_max,K_max+1))
            C_param=np.zeros(data_max)
            C_clustering=np.zeros(data_max)
            error_rf=np.zeros(data_max)
            
            #Choosing n data from generated dataset
            for n in range(start,data_max,interval):    
                print('n=',n)
                u=np.zeros((n,p))
                v=np.zeros((n,q))
                v_tilde=np.zeros((n,q+1))
                z_true=np.zeros((n,K))
                Z_true=np.zeros(n)
                y=np.zeros(n)
                
                for i in range(n):
                    u[i]=u_data[i]
                    v[i]=v_data[i]
                    v_tilde[i]=v_tilde_data[i]
                    z_true[i]=z_true_data[i]
                    Z_true[i]=Z_true_data[i]
                    y[i]=y_data[i]
                
                #Transform to DataFrame
                u_pd=pd.DataFrame(u)
                v_pd=pd.DataFrame(v)
                y_pd=pd.DataFrame(y)
                X_pd=pd.concat([u_pd,v_pd],axis=1)
            
                #Prediction by random forest
                forest_model = RandomForestRegressor(n_estimators=100)
                forest_model.fit(X_pd, y)
                forest_pred = forest_model.predict(X_new_pd)
                error_rf[n]+=(mean_squared_error(y_new, forest_pred))/D

                
                y_exp_vb_K_tilde=np.zeros((M,K_max+1))
                y_exp_vb_uniform=np.zeros(M)
                exp_L_K_tilde_uniform=np.zeros((M,K_max+1))
                exp_L_sum_uniform=np.zeros(M)
                K_estimate_uniform=np.zeros(K_max+1)
    
                
                for j in range(M):
                    for K_tilde in range(1,K_max+1):
                        #Initial value of hyper parameters
                        alpha_0=np.random.random()*10+0.05
                        pi_0=np.random.dirichlet(alpha_0*np.ones(K_tilde))
                        m_0=np.random.random((K_tilde,p)) 
                        beta_0=np.random.random()+0.001 
                        A_0=np.eye(p)
                        nu_0=p+np.random.random() 
                        mu_v0=mu_v_true  
                        Sigma_v0=Sigma_v_true 
                        mu_w0=np.zeros((K_tilde,q+1)) 
                        Lambda_w0=np.eye(q+1) 
                            
    
                        alpha=np.zeros((T,K_tilde))
                        r=np.zeros((T,n,K_tilde))
                        rho=np.zeros((T,n,K_tilde))
                        psi=np.zeros((T,M,K_tilde))
                        phi=np.zeros((T,M,K_tilde))
                        m=np.zeros((T,K_tilde,p))
                        beta=np.zeros((T,K_tilde))
                        A=np.zeros((T,K_tilde,p,p))
                        nu=np.zeros((T,K_tilde))
                        mu_w=np.zeros((T,K_tilde,q+1))
                        Lambda_w=np.zeros((T,K_tilde,q+1,q+1))                          
                        N=np.zeros((T,K_tilde))
                        x_bar=np.zeros((T,K_tilde,p))
                        S=np.zeros((T,K_tilde,p,p))
                        V=np.zeros((T,K_tilde,q+1,q+1))
                        theta=np.zeros((T,K_tilde,q+1))
                        lb=np.zeros((T,12))
                        LB=np.zeros((T,K_max+1))
                        
                        #Substituting inital values to hyper parameters
                        for k in range(K_tilde):
                            phi[0,j,k]=pi_0[k]
                            psi[0,j,k]=pi_0[k]
                            m[0]=m_0
                            beta[0,k]=beta_0
                            A[0,k]=A_0
                            nu[0,k]=nu_0
                            mu_w[0]=mu_w0
                            Lambda_w[0,k]=Lambda_w0
                            alpha[0,k]=alpha_0
                            for i in range(n):
                                rho[0,i,k]=pi_0[k]
                                r[0,i,k]=pi_0[k]
                        
                        #Variational Bayes Algorithm
                        t=0
                        while t in range(0,T-1):
                            t=t+1
                            dig=np.zeros(K_tilde)
                            for k in range(K_tilde):
                                for d in range(1,p+1):
                                    dig[k]=dig[k]+sss.digamma((nu[t-1,k]+1-d)/2)
                        #Updating r
                            for i in range(n):
                                for k in range(K_tilde):
                                    rho[t,i,k]=math.exp(0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t-1,k])))\
                                      -0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u[i]-m[t-1,k]),(1,p)),A[t-1,k]),np.reshape((u[i]-m[t-1,k]),(1,p)).T))\
                                      -0.5*(y[i]**2-2*y[i]*np.dot(mu_w[t-1,k],v_tilde[i])\
                                            +np.dot(np.dot(np.reshape(v_tilde[i],(1,q+1)),(np.linalg.inv(Lambda_w[t-1,k])+np.dot(np.reshape(mu_w[t-1,k],(1,1+q)).T,np.reshape(mu_w[t-1,k],(1,1+q))))),np.reshape(v_tilde[i],(1,q+1)).T))\
                                            /(np.square(sigma_true))\
                                            +sss.digamma(alpha[t-1,k])-sss.digamma(np.sum(alpha[t-1]))\
                                            -0.5*p*math.log(2*math.pi)-0.5*math.log(2*math.pi*np.square(sigma_true)))
                                if np.all(rho[t,i]==0) or np.any(rho[t,i]==math.inf):
                                      for k in range(K_tilde):
                                          rho[t,i,k]=math.exp(0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t-1,k]))))*\
                                          math.exp(-0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u[i]-m[t-1,k]),(1,p)),A[t-1,k]),np.reshape((u[i]-m[t-1,k]),(1,p)).T))+0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u[i]-m[t-1,0]),(1,p)),A[t-1,0]),np.reshape((u[i]-m[t-1,0]),(1,p)).T)))*\
                                          math.exp((y[i]*np.dot(mu_w[t-1,k],v_tilde[i])-y[i]*np.dot(np.sum(mu_w,axis=1)[t-1]/(K_tilde),v_tilde[i]))/(np.square(sigma_true)))*\
                                          math.exp(-0.5*(np.dot(np.dot(np.reshape(v_tilde[i],(1,q+1)),(np.linalg.inv(Lambda_w[t-1,k])+np.dot(np.reshape(mu_w[t-1,k],(1,1+q)).T,np.reshape(mu_w[t-1,k],(1,1+q))))),np.reshape(v_tilde[i],(1,q+1)).T))\
                                            /(np.square(sigma_true)))
                                      
                                if np.all(rho[t,i]==0) or np.any(rho[t,i]==math.inf):
                                      print('all zero1')
                                      print('i=',i)
                                      print(math.exp(0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t-1,k])))))
                                      print(math.exp(-0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u[i]-m[t-1,k]),(1,p)),A[t-1,k]),np.reshape((u[i]-m[t-1,k]),(1,p)).T))+0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u[i]-m[t-1,0]),(1,p)),A[t-1,0]),np.reshape((u[i]-m[t-1,0]),(1,p)).T))))
                                      print(math.exp(y[i]*np.dot(mu_w[t-1,k],v_tilde[i])/(np.square(sigma_true))))
                                      print(math.exp(-0.5*(np.dot(np.dot(np.reshape(v_tilde[i],(1,q+1)),(np.linalg.inv(Lambda_w[t-1,k])+np.dot(np.reshape(mu_w[t-1,k],(1,1+q)).T,np.reshape(mu_w[t-1,k],(1,1+q))))),np.reshape(v_tilde[i],(1,q+1)).T))\
                                        /(np.square(sigma_true))))
                                      print('//')
                                        
                            for i in range(n):
                                r[t,i]=rho[t,i]/np.sum(rho[t,i])
                                for k in range(K_tilde):
                                    if r[t,i,k]==0:
                                        r[t,i,k]=1e-300
                            
                        #Updating phi
                            for k in range(K_tilde):
                                psi[t,j,k]=math.exp(0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t-1,k])))\
                                  -0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u_new[j]-m[t-1,k]),(1,p)),A[t-1,k]),np.reshape((u_new[j]-m[t-1,k]),(1,p)).T))\
                                  +sss.digamma(alpha[t-1,k])-sss.digamma(np.sum(alpha[t-1]))\
                                  -0.5*p*math.log(2*math.pi))
                            if np.all(psi[t,j]==0)or np.any(psi[t,j]==math.inf):
                                for k in range(K_tilde):
                                    psi[t,j,k]=math.exp(0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t-1,k]))))*\
                                    math.exp(-0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u_new[j]-m[t-1,k]),(1,p)),A[t-1,k]),np.reshape((u_new[j]-m[t-1,k]),(1,p)).T))+0.5*(p/beta[t-1,0]+nu[t-1,0]*np.dot(np.dot(np.reshape((u_new[j]-m[t-1,0]),(1,p)),A[t-1,0]),np.reshape((u_new[j]-m[t-1,0]),(1,p)).T)))*\
                                    math.exp(sss.digamma(alpha[t-1,k])-sss.digamma(np.sum(alpha[t-1])))*\
                                    math.exp(-0.5*p*math.log(2*math.pi))
                            if np.all(psi[t,j]==0):
                                print('all zero2')
                                print('j=',j)
                                print(math.exp(0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t-1,k])))))
                                print(math.exp(-0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u_new[j]-m[t-1,k]),(1,p)),A[t-1,k]),np.reshape((u_new[j]-m[t-1,k]),(1,p)).T))+0.5*(p/beta[t-1,0]+nu[t-1,0]*np.dot(np.dot(np.reshape((u_new[j]-m[t-1,0]),(1,p)),A[t-1,0]),np.reshape((u_new[j]-m[t-1,0]),(1,p)).T))))
                                print(math.exp(sss.digamma(alpha[t-1,k])-sss.digamma(np.sum(alpha[t-1]))))
                                print(math.exp(-0.5*p*math.log(2*math.pi)))
              
                            phi[t,j]=psi[t,j]/np.sum(psi[t,j])
                            for k in range(K_tilde):
                                    if phi[t,j,k]==0:
                                        phi[t,j,k]=1e-300
                        
                        #Updating N
                            for k in range(K_tilde):
                                for i in range(n):
                                    N[t,k]=N[t,k]+r[t,i,k]
                                N[t,k]=N[t,k]+phi[t,j,k]
            
                        
                        #Updating x_bar
                            for k in range(K_tilde):
                                for i in range(n):
                                    x_bar[t,k]=x_bar[t,k]+(r[t,i,k]*u[i])/N[t,k]
                                x_bar[t,k]=x_bar[t,k]+(phi[t,j,k]*u_new[j])/N[t,k]
                                
                        #Updating S
                            for k in range(K_tilde):
                                for i in range(n):
                                    S[t,k]=S[t,k]+r[t,i,k]*np.dot(np.reshape((u[i]-x_bar[t,k]),(1,p)).T,np.reshape((u[i]-x_bar[t,k]),(1,p)))/N[t,k]
                                S[t,k]=S[t,k]+phi[t,j,k]*np.dot(np.reshape((u_new[j]-x_bar[t,k]),(1,p)).T,np.reshape((u_new[j]-x_bar[t,k]),(1,p)))/N[t,k]
                
                        #Upating alpha
                            for k in range(K_tilde):
                                alpha[t,k]=alpha[0,k]+N[t,k]
                    
                        
                        #Updating beta
                            for k in range(K_tilde):
                                    beta[t,k]=beta[0,k]+N[t,k]
                        
                        #Updating m
                            for k in range(K_tilde):
                                m[t,k]=(beta[0,k]*m[0,k]+N[t,k]*x_bar[t,k])/beta[t,k]
                        
                        #Updating A
                            for k in range(K_tilde):
                                A[t,k]=np.linalg.inv(np.linalg.inv(A[0,k])+N[t,k]*S[t,k]\
                                +(beta[0,k]*N[t,k])*np.dot(np.reshape((x_bar[t,k]-m[0,k]),(1,p)).T,np.reshape((x_bar[t,k]-m[0,k]),(1,p)))/(beta[0,k]+N[t,k]))
                        
                        #Updating nu
                            for k in range(K_tilde):
                                nu[t,k]=nu[0,k]+N[t,k]
                      
                        #Updating V and theta
                            for k in range(K_tilde):
                                for i in range(n):
                                    V[t,k]=V[t,k]+r[t,i,k]*np.dot(np.reshape(v_tilde[i],(1,q+1)).T,np.reshape(v_tilde[i],(1,q+1)))
                                    theta[t,k]=theta[t,k]+r[t,i,k]*y[i]*v_tilde[i]
                                      
                        #Updating Lambda_w                           
                            for k in range(K_tilde):
                                Lambda_w[t,k]=V[t,k]/(np.square(sigma_true))+Lambda_w[0,k]
                            
                        #Updating mu_w
                            for k in range(K_tilde):
                                mu_w[t,k]=np.dot(np.linalg.inv(Lambda_w[t,k]),\
                                    (theta[t,k]/(np.square(sigma_true))+np.dot(Lambda_w[0,k],mu_w[0,k])))
                                
                        #Calculating Variational Lower Bound                           
                            for k in range(K_tilde):
                                lb[t,0]=lb[t,0]+0.5*N[t,k]*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t,k]))\
                                  -p/beta[t,k]-nu[t,k]*np.trace(np.dot(S[t,k],A[t,k]))\
                                  -nu[t,k]*np.dot(np.dot(np.reshape((x_bar[t,k]-m[t,k]),(p,1)).T,A[t,k]),np.reshape((x_bar[t,k]-m[t,k]),(p,1)))\
                                  -p*math.log(2*math.pi))
                                
                                lb[t,1]=lb[t,1]-0.5*(N[t,k]*math.log(2*math.pi*sigma_true**2))
                                for i in range(n):
                                    lb[t,1]=lb[t,1]-0.5*(1/(sigma_true**2))*(r[t,i,k]*(y[i]**2-2*y[i]*np.dot(mu_w[t,k],v_tilde[i])\
                                        +np.dot(np.dot(np.reshape(v_tilde[i],(q+1,1)).T,(np.linalg.inv(Lambda_w[t,k])+np.dot(np.reshape(mu_w[t,k],(q+1,1)),\
                                                                          np.reshape(mu_w[t,k],(q+1,1)).T))),np.reshape(v_tilde[i],(q+1,1)))))
                                    
                                lb[t,2]=lb[t,2]+N[t,k]*(sss.digamma(alpha[t,k])-sss.digamma(np.sum(alpha[t])))
                                
                                gam=0
                                for d in range(p):
                                    gam=gam+math.lgamma(0.5*(nu[0,k]+1-d))
                              
                                lb[t,3]=lb[t,3]+0.5*(p*math.log(beta[0,k]/(2*math.pi))\
                                  +(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t,k])))\
                                  -p*beta[0,k]/beta[t,k]\
                                  -beta[0,k]*nu[t,k]*np.dot(np.dot(np.reshape(m[t,k]-m[0,k],(p,1)).T,A[t,k]),np.reshape(m[t,k]-m[0,k],(p,1)))\
                                  +2*(-0.5*nu[0,k]*math.log(np.linalg.det(A[0,k]))-0.5*nu[0,k]*p*math.log(2)-0.25*p*(p-1)*math.log(math.pi)-gam)\
                                  +(nu[0,k]-p-1)*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t,k])))\
                                  -(nu[0,k]*np.trace(np.dot(np.linalg.inv(A[0,k]),A[t,k]))))
                            
                            
                            gam_coef0=math.gamma(np.sum(alpha[0]))
                            gam_coeft=math.gamma(np.sum(alpha[t]))
                            for k in range(K_tilde):
                                gam_coef0=gam_coef0/math.gamma(alpha[0,k])
                                gam_coeft=gam_coeft/math.gamma(alpha[t,k])
                            lb[t,5]=math.log(gam_coef0)
                            for k in range(K_tilde):
                                lb[t,5]=lb[t,5]+(alpha[0,k]-1)*(sss.digamma(alpha[t,k])-sss.digamma(np.sum(alpha[t])))
                                
                                lb[t,6]=lb[t,6]+0.5*(math.log(np.linalg.det(Lambda_w0))\
                                  -np.trace(np.dot(Lambda_w[0,k],(np.linalg.inv(Lambda_w[t,k])+np.dot(np.reshape(mu_w[t,k],(q+1,1)),np.reshape(mu_w[t,k],(q+1,1)).T))))\
                                  +2*np.dot(np.dot(np.reshape(mu_w[t,k],(q+1,1)).T,Lambda_w[0,k]),np.reshape(mu_w[0,k],(q+1,1)))\
                                  -np.dot(np.dot(np.reshape(mu_w[0,k],(q+1,1)).T,Lambda_w[0,k]),np.reshape(mu_w[0,k],(q+1,1))))
                                
                                for i in range(n):
                                    lb[t,7]=lb[t,7]+r[t,i,k]*math.log(r[t,i,k])
                                
                                lb[t,8]=lb[t,8]+phi[t,j,k]*math.log(phi[t,j,k])
                                
                                lb[t,9]=lb[t,9]+0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t,k])))+0.5*p*math.log(beta[t,k]/(2*math.pi))\
                                -p/2-ss.wishart.entropy(nu[t,k],A[t,k])
                                
                            lb[t,10]=math.log(gam_coeft)
                            for k in range(K_tilde):
                                lb[t,10]=lb[t,10]+(alpha[t,k]-1)*(sss.digamma(alpha[t,k])-sss.digamma(np.sum(alpha[t])))
                                
                                lb[t,11]=lb[t,11]+0.5*(math.log(np.linalg.det(Lambda_w[t,k]))\
                                  -np.trace(np.dot(Lambda_w[t,k],(np.linalg.inv(Lambda_w[t,k])+np.dot(np.reshape(mu_w[t,k],(q+1,1)),np.reshape(mu_w[t,k],(q+1,1)).T))))\
                                  +np.dot(np.dot(np.reshape(mu_w[t,k],(q+1,1)).T,Lambda_w[t,k]),np.reshape(mu_w[t,k],(q+1,1))))
                                
                            LB[t,K_tilde]=lb[t,0]+lb[t,1]+lb[t,2]+lb[t,3]+lb[t,4]+lb[t,5]+lb[t,6]-lb[t,7]-lb[t,8]-lb[t,9]-lb[t,10]-lb[t,11]+math.log(math.factorial(K_tilde))
                            
                            if abs(LB[t,K_tilde]-LB[t-1,K_tilde])<0.0001:
                                break
                    #End of Variational Bayes algorithm
                    
                      
                    #Caluculating prediction                                      
                        for k in range(K_tilde):
                            y_exp_vb_K_tilde[j,K_tilde]=y_exp_vb_K_tilde[j,K_tilde]+phi[t,j,k]*np.dot((mu_w[t,k]),v_tilde_new[j])
                        
                        exp_L_K_tilde_uniform[j,K_tilde]=math.exp(LB[t,K_tilde])*STD
                        
                        y_exp_vb_uniform[j]=y_exp_vb_uniform[j]+exp_L_K_tilde_uniform[j,K_tilde]*y_exp_vb_K_tilde[j,K_tilde]
                        exp_L_sum_uniform[j]=exp_L_sum_uniform[j]+exp_L_K_tilde_uniform[j,K_tilde]
                    
                    y_exp_vb_uniform[j]=y_exp_vb_uniform[j]/exp_L_sum_uniform[j]
                    
                    for K_tilde in range(1,K_max+1):
                        K_estimate_uniform[K_tilde]=K_estimate_uniform[K_tilde]+exp_L_K_tilde_uniform[j,K_tilde]/exp_L_sum_uniform[j]
                        error_vb_K_tilde[n,K_tilde]=error_vb_K_tilde[n,K_tilde]+((y_exp_vb_K_tilde[j,K_tilde]-y_new[j])**2)/M
                    error_vb_uniform[n]=error_vb_uniform[n]+((y_exp_vb_uniform[j]-y_new[j])**2)/M
                    
                for K_tilde in range(1,K_max+1):
                    print('K_tilde=',K_tilde)
                    print('error_vb_K_tilde=',error_vb_K_tilde[n,K_tilde])
                    K_prob_uniform[n,K_tilde]=K_estimate_uniform[K_tilde]/M
                    print('K_prob_uniform=',K_prob_uniform[n,K_tilde])
                print("\n")
                print('error_vb_uniform=',error_vb_uniform[n])
                print('param_dist=',C_param[n])
                print('cluster_dist=',C_clustering[n])
        
                        
               
           
                
            #Record to file
            f_vb_uniform=open("data_vb_uniform.txt","a")
            f_vb_1=open("data_vb_1.txt","a")
            f_vb_2=open("data_vb_2.txt","a")
            f_vb_3=open("data_vb_3.txt","a")
            f_vb_4=open("data_vb_4.txt","a")
            f_vb_5=open("data_vb_5.txt","a")
            f_rf=open("data_rf.txt","a")
            f_cluster_uniform=open("data_cluster_uniform.txt","a")
            f_vb_uniform.write(str(param_iter))
            f_vb_1.write(str(param_iter))
            f_vb_2.write(str(param_iter))
            f_vb_3.write(str(param_iter))
            f_vb_4.write(str(param_iter))
            f_vb_5.write(str(param_iter))
            f_rf.write(str(param_iter))
            f_cluster_uniform.write(str(param_iter))
            for n in range(start,data_max,interval):
                f_vb_uniform.write(","+str(error_vb_uniform[n]))
                f_vb_1.write(","+str(error_vb_K_tilde[n,1]))           
                f_vb_2.write(","+str(error_vb_K_tilde[n,2]))
                f_vb_3.write(","+str(error_vb_K_tilde[n,3]))
                f_vb_4.write(","+str(error_vb_K_tilde[n,4]))
                f_vb_5.write(","+str(error_vb_K_tilde[n,5]))
                f_rf.write(","+str(error_rf[n]))
                for K_tilde in range(1,K_max+1):
                    f_cluster_uniform.write(","+str(K_prob_uniform[n,K_tilde]))
            f_vb_uniform.write("\n")
            f_vb_1.write("\n")
            f_vb_2.write("\n")
            f_vb_3.write("\n")
            f_vb_4.write("\n")
            f_vb_5.write("\n")
            f_rf.write("\n")
            f_cluster_uniform.write("\n")
            f_vb_uniform.close()
            f_vb_1.close()
            f_vb_2.close()
            f_vb_3.close()
            f_vb_4.close()
            f_vb_5.close()
            f_rf.close()
            f_cluster_uniform.close()


