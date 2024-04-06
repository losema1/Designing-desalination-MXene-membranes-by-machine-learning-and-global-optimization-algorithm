# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:51:51 2022

@author: maxuanchao
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
import pandas as pd
 
class PSO:
    def __init__(self, parameters):
        """
        particle swarm optimization
        """
        # initialization
        self.NGEN = parameters[0]    
        self.pop_size = parameters[1]   
        self.var_num = len(parameters[2])     # Number of variables
        self.bound = []                 # variable constraint range
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
 
        self.pop_x = np.zeros((self.pop_size, self.var_num))    # The positions of all particles
        # self.pop_v = np.zeros((self.pop_size, self.var_num))    # speed of all particles
        self.p_best = np.zeros((self.pop_size, self.var_num))   # The optimal position of each particle
        self.g_best = np.zeros((1, self.var_num))   # Global optimal position
 
        # Initialize the initial global optimal solution of generation 0
        temp = -1
        flag=0
        self.pop_v=[]
        self.gen=-1
        for i in range(self.pop_size):
            pop_x1=[]
            popv=[]
            for j in range(self.var_num):
                if j==0 or j==3 or j==5:
                    pop_x1.append(random.randint(self.bound[0][j], self.bound[1][j]))  # initialise popluation
                if j==1 or j==4 or j==2:
                    pop_x1.append(round(random.uniform(self.bound[0][j], self.bound[1][j]),2))
                # self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                if j<6:
                    popv.append(random.uniform(0, 1))
            self.pop_v.append(popv)
            if flag==0:
                df=pd.DataFrame(pop_x1).T
                df.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
            if flag==1:
                df.loc[len(df)]=pop_x1
            flag=1
            fitness,flux_0,reject_0 = self.fitness(df)
            self.pop_x[i]=pop_x1
            self.p_best[i] = self.pop_x[i]     # Store the best individuals
            if fitness[i] > temp:
                self.g_best = self.p_best[i]
                temp = fitness[i]
        self.pop_v=np.array(self.pop_v)
        result_0=[flux_0,reject_0,fitness]        
        self.write_data('0',df,result_0)

    def write_data(self,name,df_1,df_2):
        df_result=pd.DataFrame(df_2).T
        df_result.columns = ['flux','rejection','fitness']
        df_all=pd.concat([df_1,df_result],axis=1,join='inner')

        df_all.to_csv(name+'.csv',index=False)
        return print("Exported successfully")
    
    def fitness(self, df):
        """
        fitness function
        """
        # load model from file

        if len(df)==6 and self.gen !=-1 and np.array(df).ndim==1:
            df=pd.DataFrame(df).T
            df.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
        fit_value=[]
        loaded_model_reject = pickle.load(open("reject.dat", "rb"))
        loaded_model_flux = pickle.load(open("flux.dat", "rb"))
        # make predictions for test data
        y_reject = loaded_model_reject.predict(df)
        y_flux = loaded_model_flux.predict(df)
        pre_flux = [round(value,1) for value in y_flux]
        pre_reject = [round(value,1) for value in y_reject]
        for j in range(len(pre_reject)):
            if pre_reject[j]>100:
                pre_reject[j]=100
        for i in range(len(y_flux)):
            fit_value.append((pre_flux[i]**0.5)*(pre_reject[i]**8))
        predictions=[round(value,1) for value in fit_value]       
        return predictions,pre_flux,pre_reject 

    
    
    def update_operator(self, pop_size):
        """
       Update operator: update the position and velocity at the next moment
        """
        c1 = 2     # Set acceleration factor
        c2 = 2
        w = 0.4    # Set inertia weight 
        
        flag=0
        for i in range(pop_size):
            # Update particle speed
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                        self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            # Update particle position
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越cross-border protection
            for j in range(self.var_num):
                if j==5 or j==0:
                   self.pop_x[i][j]=round(self.pop_x[i][j]) 
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]

            pop_x2=self.pop_x[i]
            local_best=self.p_best[i]
            if flag==0:
                df=pd.DataFrame(pop_x2).T
                df_best=pd.DataFrame(local_best).T
                df.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
                df_best.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
            if flag==1:
                df.loc[len(df)]=pop_x2
                df_best.loc[len(df)]=local_best
            flag=1
            fitness,flux_0,reject_0 = self.fitness(df)
            fitness_best,flux_best,reject_best = self.fitness(df_best)
            #Update p_best and g_best
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i])> self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        
        
        result_main=[flux_best,reject_best,fitness_best]        
        self.write_data("{}".format(self.gen+1),df_best,result_main)
        
    def main(self):
        popobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        self.ng_best=self.pop_x[0]
        for gen in range(self.NGEN):
            self.gen=gen
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best)[0][0])
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best)[0][0] > self.fitness(self.ng_best)[0][0]:
                self.ng_best = self.g_best.copy()
            print('Best position：{}'.format(self.ng_best))
            print('Maximum function value:{}'.format(self.fitness(self.ng_best)))
        print("---- End of (successful) Searching ----")
 
        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()
 
 

NGEN = 100 # Set the number of iterations
popsize = 100  # Set population size
up = [4, 120, 1.2, 350,1.2,2]  # upper range for variables
low = [1, 10, 0.4, 10,0.3,1]  # lower range for variables
parameters = [NGEN, popsize, low, up]
pso = PSO(parameters)
pso.main()