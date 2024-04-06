# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 22:23:06 2022

@author: ma
"""


import random
from operator import itemgetter
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Initialize the gene and get the size
class Gene:
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size=len(data['data'])


class GA:
    """
    This is a class of GA algorithm.
    """
 
    def __init__(self, parameter):
        """
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value.
        The data structure of pop is composed of several individuals which has the form like that:
            {'Gene':a object of class Gene, 'fitness': Derived from machine learning models}
        Representation of Gene is a list: ['Material','size','Charge coefficient','pressure','concentration','shape']
 
        """
        # parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
        self.parameter = parameter
 
        low = self.parameter[4]
        up = self.parameter[5]
 
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)
 
        pop = []
        flag=0
        for i in range(self.parameter[3]*5):
            geneinfo = []
            for pos in range(len(low)):
                if pos==0 or pos==3 or pos==5:
                    geneinfo.append(random.randint(self.bound[0][pos], self.bound[1][pos]))  # initialise popluation
                if pos==1 or pos==4 or pos==2:
                    geneinfo.append(round(random.uniform(self.bound[0][pos], self.bound[1][pos]),1))  # initialise popluation
            if flag==0:
                df=pd.DataFrame(geneinfo).T
                df.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
            if flag==1:
                df.loc[len(df)]=geneinfo
            flag=1
#            self.write_data(geneinfo)           
            self.geneinfo=geneinfo
            self.df=df
            fitness,flux_0,reject_0 = self.evaluate(df)  # evaluate each chromosome
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness[i],  'flux':flux_0[i], 'rejection':reject_0[i]})  # store the chromosome and its fitness
        result_0=[flux_0,reject_0,fitness]
#        df_result_0=pd.DataFrame(result_0).T
#        df_result_0.columns = ['flux','rejection','fitness']
#        df_all=pd.concat([df,df_result_0],axis=1,join='inner')
        self.write_data('0',df,result_0)
        self.df=df
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population
        
    def evaluate(self, df):
        """
        fitness function
        """
        # load model from file
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

    

 
    def selectBest(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)          # from large to small, return a pop
        return s_inds[0]
    
    def write_data(self,name,df_1,df_2):
        
        df_result=pd.DataFrame(df_2).T
        df_result.columns = ['flux','rejection','fitness']
        df_all=pd.concat([df_1,df_result],axis=1,join='inner')

        df_all.to_csv(name+'.csv',index=False)
        return print("Export successful")
     
    
    def selection(self, individuals, k):
        """
        Choose the one with the highest fitness ranking
        """
        s_inds = sorted(individuals, key=itemgetter("fitness"),reverse=True)  # sort the pop by the reference of fitness

        sum_fits = sum(ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits], as threshold
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']  # sum up the fitness
                if sum_ >= u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of
                    # [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    random.shuffle(s_inds)
                    break
        # from small to large, due to list.pop() method get the last element
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=True)
        return chosen
 #######################################################################################################
        # s_inds = sorted(individuals, key=itemgetter("fitness"),reverse=True)
        # chosen=[]
        # self.s_inds=s_inds
        # for i in s_inds:
        #     if len(chosen)<k:
        #         if i['rejection']>97.5:
        #             chosen.append(i)
        # return chosen
        
#                         
    def crossoperate(self, offspring):
        """
        cross operation
        here we use two points crossoperate
        for example: gene1: [7, 2, 4, 7], gene2: [3, 6, 8, 2], if pos1=1, pos2=2
        7 | 2 | 4  7
        3 | 6 | 8  2
        =
        3 | 2 | 8  2
        7 | 6 | 4  7
        
        """
        dim = len(offspring[0]['Gene'].data)
 
        geninfo1 = offspring[0]['Gene'].data   # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data   # Gene's data of second offspring chosen from the selected pop
 
        if dim == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.choice([0,1,2,3,4,5 ])  # select a position in the range from 0 to dim-1,
            pos2 = random.choice([0,1,2,3,4,5 ])
 
        newoff1 = Gene(data=[])  # offspring1 produced by cross operation
        newoff2 = Gene(data=[])  # offspring2 produced by cross operation
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
            # if i==pos1:
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2
        self.temp1=temp2
 
        return newoff1, newoff2
 
    def mutation(self, crossoff, bound):
        """
        mutation operation
        """
        dim = len(crossoff.data)
 
        if dim == 1:
            pos = 0
        else:
            pos = random.randrange(0, dim)  # chose a position in crossoff to perform mutation.
 
#        crossoff.data[pos] = random.randint(bound[0][pos], bound[1][pos])
        
        if pos==0 or pos==3 or pos==5:
            crossoff.data[pos] = random.randint(self.bound[0][pos], self.bound[1][pos])
        if pos==1 or pos==4 or pos==2:
            crossoff.data[pos]=round(random.uniform(self.bound[0][pos], self.bound[1][pos]),1)

        return crossoff
 
    def GA_main(self):
        """
        main frame work of GA
        """
        popsize = self.parameter[3]
        best_select=[]
 
        print("Start of evolution")
        # chosen_num=30   
        
        # Begin the evolution
        for g in range(NGEN):
 
            print("############### Generation {} ###############".format(g))
            chosen_num=60-10*g
            if chosen_num<5:
                chosen_num=5
            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, chosen_num)
            count=0
            flag_main=0
            nextoff = []
            select_gen=[]
            while len(nextoff) < popsize:
                # Apply crossover and mutation on the offspring
 
                # Select two individuals
#                offspring = [selectpop.pop() for _ in range(2)]  
                # offspring = random.sample(selectpop,2)
                offspring = self.selection(selectpop, 2)
                self.offe= offspring
                if random.random() < CXPB:  # cross two individuals with probability CXPB
                    # geninfo_group=self.charge_code(offspring)
                    crossoff_1, crossoff_2 = self.crossoperate(offspring)
                    crossoff1=crossoff_1
                    self.off= crossoff1
                    crossoff2=crossoff_2
                    if random.random() < MUTPB:  # mutate an individual with probability MUTPB
                        muteoff1 = self.mutation(crossoff1, self.bound)
                        muteoff2 = self.mutation(crossoff2, self.bound)
                        if flag_main==0:
                            df_main=pd.DataFrame(muteoff1.data).T                            
                            df_main.loc[len(df_main)]=muteoff2.data
                            df_main.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
                        if flag_main==1:
                            df_main.loc[len(df_main)]=muteoff1.data
                            df_main.loc[len(df_main)]=muteoff2.data
                        flag_main=1   
                        fitness_main,flux_main,reject_main= self.evaluate(df_main)
                        nextoff.append({'Gene': muteoff1, 'fitness': fitness_main[2*count],'flux': flux_main[2*count],'rejection': reject_main[2*count]})
                        nextoff.append({'Gene': muteoff2, 'fitness': fitness_main[2*count+1],'flux': flux_main[2*count+1],'rejection': reject_main[2*count]+1})
                    else:
                        if flag_main==0:
                            df_main=pd.DataFrame(crossoff1.data).T                             
                            df_main.loc[len(df_main)]=crossoff2.data
                            df_main.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
                        if flag_main==1:
                            df_main.loc[len(df_main)]=crossoff1.data
                            df_main.loc[len(df_main)]=crossoff2.data
                        flag_main=1   
                        fitness_main,flux_main,reject_main= self.evaluate(df_main)
                        nextoff.append({'Gene': crossoff1, 'fitness': fitness_main[2*count],'flux': flux_main[2*count],'rejection': reject_main[2*count]})
                        nextoff.append({'Gene': crossoff2, 'fitness': fitness_main[2*count+1],'flux': flux_main[2*count+1],'rejection': reject_main[2*count]+1})
                else:
                    if flag_main==0:
                        df_main=pd.DataFrame(offspring[0]['Gene'].data).T                             
                        df_main.loc[len(df_main)]=offspring[1]['Gene'].data
                        df_main.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
                    if flag_main==1:
                        df_main.loc[len(df_main)]=offspring[0]['Gene'].data
                        df_main.loc[len(df_main)]=offspring[1]['Gene'].data
                    flag_main=1
                    fitness_main,flux_main,reject_main= self.evaluate(df_main)                    
                    nextoff.append({'Gene': Gene(data=offspring[0]['Gene'].data), 'fitness': offspring[0]['fitness'],'flux': offspring[0]['flux'],'rejection':offspring[0]['rejection']})
                    nextoff.append({'Gene': Gene(data=offspring[1]['Gene'].data), 'fitness': offspring[1]['fitness'],'flux': offspring[1]['flux'],'rejection':offspring[1]['rejection']})
            # The population is entirely replaced by the offspring
            self.pop = nextoff
            self.cross=crossoff1.data
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            #Record genetic information for each generation
            result_main=[flux_main,reject_main,fitness_main]
            
            self.write_data("{}".format(g+1),df_main,result_main)
            best_ind = self.selectBest(self.pop)
 
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
            best_select.append(max(fits))
            print("Best individual found is {}, {}".format(self.bestindividual['Gene'].data,
                                                           self.bestindividual['fitness']))
            print("  Max fitness of current pop: {}".format(max(fits)))
 
        print("------ End of (successful) evolution ------")
        plt.figure()
        plt.title("Best Fitness")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(self.parameter[2])]
        plt.plot(t,best_select, color='b', linewidth=2)
        plt.show()
 
 
#if __name__ == "__main__":
#The four parameters are: crossover probability, mutation probability, number of iterations and population size.
CXPB, MUTPB, NGEN, popsize = 0.35, 0.2, 100, 200  # popsize must be even number
#The upper and lower limits of the parameters are material, pore size, charge coefficient (TI1, TI0, C), pressure, concentration, and shape respectively.
up = [4, 150, 1.2, 350,1.2,2]  # upper range for variables
low = [1, 10, 0.4, 10,0.2,1]  # lower range for variables
#Input parameters
parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
#Instantiate
run = GA(parameter)
pops=run.pop
aa=run.geneinfo
bb=run.df
bb.columns = ['Material','size','Charge coefficient','pressure','concentration','shape']
run.GA_main()



