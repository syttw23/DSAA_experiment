# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:19:22 2019

@author: Haruka Murayama
"""




import numpy as np
import scipy.stats as ss
import scipy.special as sss
import math

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets


#Constants
p=1
q=6
K_max=5
T=30
R=2.
STD=1e180
DATA_MAX=60
START=10
INTERVAL=10
M=50
D=50


diabetes_data = datasets.load_diabetes()
diabetes_df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
diabetes_df['target'] = diabetes_data.target
diabetes=diabetes_df.drop('sex',axis=1)



feature_names_u = ["bp"]
#change feature_name_u anonymously
feature_names_v=["s1", "s2","s3","s4","s5","s6"]


for data_iter in range(D):
    print('data_iter=',data_iter)
    error_rf=np.zeros(DATA_MAX)
    error_vb_uniform=np.zeros(DATA_MAX)
    for n in range(START,DATA_MAX, INTERVAL): 
        print("n=",n)      
        diabetes_sampled=diabetes.sample(n+M)
        y=diabetes_sampled.target   
        X=diabetes_sampled[np.concatenate((feature_names_u,feature_names_v))]



        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=M, train_size=n)
        
        
        u=train_X[feature_names_u]
        v=train_X[feature_names_v]
        
        
        
        u_new=val_X[feature_names_u]
        v_new=val_X[feature_names_v]
        u_and_v=pd.concat([u,v],axis=1)
        u_and_v_new=pd.concat([u_new,v_new],axis=1)
        
        
        
        #prediction by random_forest
        forest_model = RandomForestRegressor(n_estimators=100)
        forest_model.fit(u_and_v, train_y)
        forest_pred = forest_model.predict(u_and_v_new)
        error_rf[n]=mean_squared_error(val_y, forest_pred)
        print('error_rf=',error_rf[n])
        
        

        mean_y=train_y.mean()
        variance_y=train_y.std(ddof=0)
        train_y=(train_y-train_y.mean())/train_y.std(ddof=0)
        
        # #preprocessing
        # v=(v - v.mean())/v.std(ddof=0)
        # v_new=(v_new-v_new.mean())/v_new.std(ddof=0)
        # u=(u - u.mean())/u.std(ddof=0)
        # u_new=(u_new-u_new.mean())/u_new.std(ddof=0)
        # train_y=(train_y-train_y.mean())/train_y.std(ddof=0)
        
        
        
        y_exp_vb_K_tilde=np.zeros((M,K_max+1))
        y_exp_vb_uniform=np.zeros(M)
        exp_L_K_tilde_uniform=np.zeros((M,K_max+1))
        exp_L_sum_uniform=np.zeros(M)
        K_estimate_uniform=np.zeros(K_max+1)
        error_vb_K_tilde=np.zeros(K_max+1)
        K_prob_uniform=np.zeros(K_max+1)
        
        u=u.values
        v=v.values
        v_tilde=np.zeros([n,q+1])
        v_tilde_new=np.zeros([M,q+1])
        for c in range(n):
            v_tilde[c]=np.insert(v[c],0,[1])
        u_new=u_new.values
        v_new=v_new.values
        for j in range(M):
            v_tilde_new[j]=np.insert(v_new[j],0,[1])
        y=train_y.values
        y_new=val_y.values
        
        
        
        for j in range(M):
            for K_tilde in range(1,K_max+1):

                alpha_0=np.random.random()*10+0.05
                pi_0=np.random.dirichlet(alpha_0*np.ones(K_tilde))
                m_0=np.random.random((K_tilde,p))
                beta_0=np.random.random()+0.001  
                A_0=np.eye(p) 
                nu_0=p+np.random.random()  
                mu_v0=np.zeros(q)  
                Sigma_v0=np.eye(q) 
                mu_w0=np.zeros((K_tilde,q+1))
                Lambda_w0=np.eye(q+1)
                sigma_true=0.5
                    
                    

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
                

                t=0
                while t in range(0,T-1):
                    t=t+1
                    dig=np.zeros(K_tilde)
                    for k in range(K_tilde):
                        for d in range(1,p+1):
                            dig[k]=dig[k]+sss.digamma((nu[t-1,k]+1-d)/2)
               
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
                              print('t=',t)
                              print('i=',i)
                              print(math.exp(0.5*(dig[k]+p*math.log(2)+math.log(np.linalg.det(A[t-1,k])))))
                              print(math.exp(-0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u[i]-m[t-1,k]),(1,p)),A[t-1,k]),np.reshape((u[i]-m[t-1,k]),(1,p)).T))+0.5*(p/beta[t-1,k]+nu[t-1,k]*np.dot(np.dot(np.reshape((u[i]-m[t-1,0]),(1,p)),A[t-1,0]),np.reshape((u[i]-m[t-1,0]),(1,p)).T))))
                              print(math.exp(y[i]*np.dot(mu_w[t-1,k],v_tilde[i])/(np.square(sigma_true))))
                              print(math.exp(-0.5*(np.dot(np.dot(np.reshape(v_tilde[i],(1,q+1)),(np.linalg.inv(Lambda_w[t-1,k])+np.dot(np.reshape(mu_w[t-1,k],(1,1+q)).T,np.reshape(mu_w[t-1,k],(1,1+q))))),np.reshape(v_tilde[i],(1,q+1)).T))\
                                /(np.square(sigma_true))))
                              print('//')
                              print(-0.5*(np.dot(np.dot(np.reshape(v_tilde[i],(1,q+1)),(np.linalg.inv(Lambda_w[t-1,k])+np.dot(np.reshape(mu_w[t-1,k],(1,1+q)).T,np.reshape(mu_w[t-1,k],(1,1+q))))),np.reshape(v_tilde[i],(1,q+1)).T))\
                                /(np.square(sigma_true)))
                                
                    for i in range(n):
                        r[t,i]=rho[t,i]/np.sum(rho[t,i])
                        for k in range(K_tilde):
                            if r[t,i,k]==0:
                                r[t,i,k]=1e-300
                    
                
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
                
                
                    for k in range(K_tilde):
                        for i in range(n):
                            N[t,k]=N[t,k]+r[t,i,k]
                        N[t,k]=N[t,k]+phi[t,j,k]
        
                
                
                    for k in range(K_tilde):
                        for i in range(n):
                            x_bar[t,k]=x_bar[t,k]+(r[t,i,k]*u[i])/N[t,k]
                        x_bar[t,k]=x_bar[t,k]+(phi[t,j,k]*u_new[j])/N[t,k]
                        
               
                    for k in range(K_tilde):
                        for i in range(n):
                            S[t,k]=S[t,k]+r[t,i,k]*np.dot(np.reshape((u[i]-x_bar[t,k]),(1,p)).T,np.reshape((u[i]-x_bar[t,k]),(1,p)))/N[t,k]
                        S[t,k]=S[t,k]+phi[t,j,k]*np.dot(np.reshape((u_new[j]-x_bar[t,k]),(1,p)).T,np.reshape((u_new[j]-x_bar[t,k]),(1,p)))/N[t,k]
        
                
                    for k in range(K_tilde):
                        alpha[t,k]=alpha[0,k]+N[t,k]
            
                
                
                    for k in range(K_tilde):
                            beta[t,k]=beta[0,k]+N[t,k]
                
                
                    for k in range(K_tilde):
                        m[t,k]=(beta[0,k]*m[0,k]+N[t,k]*x_bar[t,k])/beta[t,k]
                
                
                    for k in range(K_tilde):
                        A[t,k]=np.linalg.inv(np.linalg.inv(A[0,k])+N[t,k]*S[t,k]\
                        +(beta[0,k]*N[t,k])*np.dot(np.reshape((x_bar[t,k]-m[0,k]),(1,p)).T,np.reshape((x_bar[t,k]-m[0,k]),(1,p)))/(beta[0,k]+N[t,k]))
                
                
                    for k in range(K_tilde):
                        nu[t,k]=nu[0,k]+N[t,k]
              
                
                    for k in range(K_tilde):
                        for i in range(n):
                            V[t,k]=V[t,k]+r[t,i,k]*np.dot(np.reshape(v_tilde[i],(1,q+1)).T,np.reshape(v_tilde[i],(1,q+1)))
                            theta[t,k]=theta[t,k]+r[t,i,k]*y[i]*v_tilde[i]
            
            
                
                
                    for k in range(K_tilde):
                        Lambda_w[t,k]=V[t,k]/(np.square(sigma_true))+Lambda_w[0,k]
                    
                
                    for k in range(K_tilde):
                        mu_w[t,k]=np.dot(np.linalg.inv(Lambda_w[t,k]),\
                            (theta[t,k]/(np.square(sigma_true))+np.dot(Lambda_w[0,k],mu_w[0,k])))
                        
                 
                    
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
            
            
              
        
          
                
                for k in range(K_tilde):
                    y_exp_vb_K_tilde[j,K_tilde]=y_exp_vb_K_tilde[j,K_tilde]+phi[t,j,k]*np.dot((mu_w[t,k]),v_tilde_new[j])
          
                
                exp_L_K_tilde_uniform[j,K_tilde]=math.exp(LB[t,K_tilde])*STD
                
                y_exp_vb_uniform[j]=y_exp_vb_uniform[j]+exp_L_K_tilde_uniform[j,K_tilde]*y_exp_vb_K_tilde[j,K_tilde]
                exp_L_sum_uniform[j]=exp_L_sum_uniform[j]+exp_L_K_tilde_uniform[j,K_tilde]
            
            y_exp_vb_uniform[j]=y_exp_vb_uniform[j]/exp_L_sum_uniform[j]
         
            
            for K_tilde in range(1,K_max+1):
                K_estimate_uniform[K_tilde]=K_estimate_uniform[K_tilde]+exp_L_K_tilde_uniform[j,K_tilde]/exp_L_sum_uniform[j]
                error_vb_K_tilde[K_tilde]=error_vb_K_tilde[K_tilde]+((y_exp_vb_K_tilde[j,K_tilde]-y_new[j])**2)/M
         
        y_exp_vb_uniform=y_exp_vb_uniform*variance_y+mean_y
        error_vb_uniform[n]=mean_squared_error(val_y, y_exp_vb_uniform)
            
        for K_tilde in range(1,K_max+1):

            K_prob_uniform[K_tilde]=K_estimate_uniform[K_tilde]/M
        print('error_vb_uniform=',error_vb_uniform[n])
        print('error_rf-error_vb_uniform=',error_rf[n]-error_vb_uniform[n])
        
        
        
           
            
 
    f_vb_uniform=open("data_vb_uniform.txt","a")
    f_rf=open("data_rf.txt","a")
    for n in range(START,DATA_MAX,INTERVAL):
        f_vb_uniform.write(","+str(error_vb_uniform[n]))
        f_rf.write(","+str(error_rf[n]))
    f_vb_uniform.write("\n")
    f_rf.write("\n")
    f_vb_uniform.close()
    f_rf.close()


