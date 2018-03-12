"""""
Author: Vahid Saberi
Date: March 11, 2018

Description:

This is a simulation of syphilis transmission/prevalence
in the population of men who have sex with men (MSM) in the
city of Toronto. The purpose of this simulation is to find the
optimal screening strategy and the threshold rate of screening
required to locally eliminate syphilis.

"""

#import required moduls
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.optimize as opt



class Syphilis_dynamics:

    """
    This class is implemented to define compartmental dynamic model of syphilis transmission
    for a subpopulation with certain rate of partner change.

    Model parameters:

    name: subpopulation name (e.g. low activity, high activity, etc.)
    miu:        Entry/exit rate (per capita per year)
    nu:         Proportion of individuals in 2 stage who progress to early latent stage
    tau_p:      Average treatment rate for primary stage (per capita per year)
    tau_s:      Average treatment rate for secondary stage (per capita per year)
    tau_el:     Average treatment rate for early latent stage (per capita per year)
    tau_rs:     Average treatment rate for recurrent secondary stage (per capita per year)
    tau_L:      Average treatment rate for latent stage (per capita per year)
    tau_t:      Average treatment rate for tertiary stage (per capita per year)
    gamma:      Average rate of loss of immunity (per capita per year)
    sigma1:     Average rate of progression from exposed to primary stage (per capita per year)
    sigma2:     Average rate of progression from primary to secondary stage (per capita per year)
    sigma3:     Average rate of progression from secondary to latent (early or late) stage (per capita per year)
    sigma4:     Average rate of progression from early latent to recurrent secondary stage (per capita per year)
    sigma5:     Average rate of progression from latent to tertiary stage (per capita per year)
    C:          Number of partnerships (per year)



    Class methods inputs:



    X0:         Initial population fraction of susceptible hosts
    E0:         Initial population fraction of exposed people
    YP0:        Initial population fraction of people with primary syphilis
    YS0:        Initial population fraction of people with secondary syphilis
    EL0:        Initial population fraction of people with early latent syphilis
    YRS0:       Initial population fraction of people with recurrent secondary syphilis
    L0:         Initial population fraction of people with latent syphilis
    Yt0:        Initial population fraction of people with tertiary syphilis
    Z0:         Initial population fraction of people with partial (temporary) immunity


    Class potential outputs:

    time:       Time series (year)
    X:          population fraction of susceptible hosts (time series)
    E:          population fraction of exposed people (time series)
    YP:         population fraction of people with primary syphilis (time series)
    YS:         population fraction of people with secondary syphilis (time series)
    EL:         population fraction of people with early latent syphilis (time series)
    YRS:        population fraction of people with recurrent secondary syphilis (time series)
    L:          population fraction of people with latent syphilis (time series)
    Yt:         population fraction of people with tertiary syphilis (time series)
    Z:          population fraction of people with partial (temporary) immunity (time series)


    """


    def __init__(self,
                 name,
                 miu,
                 nu,
                 tau_p,
                 tau_s,
                 tau_el,
                 tau_rs,
                 tau_L,
                 tau_t,
                 gamma,
                 sigma1,
                 sigma2,
                 sigma3,
                 sigma4,
                 sigma5,
                 C
                 ):
        self.name=name
        self.miu=miu
        self.nu=nu
        self.tau_p=tau_p
        self.tau_s=tau_s
        self.tau_el=tau_el
        self.tau_rs=tau_rs
        self.tau_L=tau_L
        self.tau_t=tau_t
        self.gamma=gamma
        self.sigma1=sigma1
        self.sigma2=sigma2
        self.sigma3=sigma3
        self.sigma4=sigma4
        self.sigma5=sigma5
        self.C=C


        #empty states initialize
        # initialize
        self.time = []
        self.X = []
        self.E = []
        self.YP = []
        self.YS = []
        self.EL = []
        self.YRS = []
        self.L = []
        self.Yt = []
        self.Z = []
        self.incidence = []



    def set_initial_states(self,X0,E0,YP0,YS0,EL0,YRS0,L0,Yt0,Z0):
        self.initial_states=[]
        self.initial_states[:]=[X0,E0,YP0,YS0,EL0,YRS0,L0,Yt0,Z0]


    def states_dot(self,states,lambda_):
        X=states[0]
        E=states[1]
        YP=states[2]
        YS=states[3]
        EL=states[4]
        YRS=states[5]
        L=states[6]
        Yt=states[7]
        Z=states[8]
        N=X+E+YP+YS+EL+YRS+L+Yt+Z

        #empty states_dot list
        states_dot=[]

        #calculate states_dot
        states_dot.append(self.miu*N+self.tau_p*YP+self.tau_s*YS+self.gamma*Z-(lambda_+self.miu)*X)
        states_dot.append(lambda_*X-(self.sigma1+self.miu)*E)
        states_dot.append(self.sigma1*E-(self.miu+self.tau_p+self.sigma2)*YP)
        states_dot.append(self.sigma2*YP-(self.miu+self.tau_s+self.sigma3)*YS)
        states_dot.append(self.nu*self.sigma3*YS-(self.miu+self.tau_el+self.sigma4)*EL)
        states_dot.append(self.sigma4*EL-(self.miu+self.tau_rs+self.sigma3)*YRS)
        states_dot.append(self.sigma3*YRS+(1-self.nu)*self.sigma3*YS-(self.sigma5+self.miu+self.tau_L)*L)
        states_dot.append(self.sigma5*L-(self.miu+self.tau_t)*Yt)
        states_dot.append(self.tau_L*L+self.tau_t*Yt+self.tau_rs*YRS+self.tau_el*EL-(self.miu+self.gamma)*Z)

        return states_dot


    def set_states(self,states,time):

        if not (len(self.time)==0):
            time=[self.time[-1]+time[i] for i in range(len(time))]
        self.time.extend(time)
        self.X.extend(states[0][:])
        self.E.extend(states[1][:])
        self.YP.extend(states[2][:])
        self.YS.extend(states[3][:])
        self.EL.extend(states[4][:])
        self.YRS.extend(states[5][:])
        self.L.extend(states[6][:])
        self.Yt.extend(states[7][:])
        self.Z.extend(states[8][:])
        self.incidence[:]=[self.YP[i]+self.YS[i]+self.EL[i]+self.YRS[i]+self.L[i]+self.Yt[i] for i in range(len(self.YP))]
        self.initial_states[:]=[states[i][-1] for i in range(len(states))]


    def apply_screening_strategy(self,
                                 tau_p,
                                 tau_s,
                                 tau_el,
                                 tau_rs,
                                 tau_L,
                                 tau_t):
        self.tau_p = tau_p
        self.tau_s = tau_s
        self.tau_el = tau_el
        self.tau_rs = tau_rs
        self.tau_L = tau_L
        self.tau_t = tau_t

    def bar_plot_incidence(self):
        values = [self.YP[-1], self.YS[-1], self.EL[-1], self.YRS[-1], self.L[-1], self.Yt[-1]]
        labels = ['YP', 'YS', 'EL', 'YRS', 'L', 'Yt']
        plt.bar(labels, values)
        plt.ylabel('syphilis incidence per 100000')
        plt.title(self.name)
        plt.show()

class Syphilis_simulation:
    """
        This class is implemented to simulate syphilis transmission between several subgroups with different rate of sexual activity.
        It takes a list of subgroups with certain rate of partner change and simulates their interactions.

        Model parameters:

        groups:         A list of subpopulations with different partnership rate
        epsilon:        Mixing factor (0 to 1) for fully random (0) to fully assortative (1)
        beta:           Syphilis biological probability of transmission per partnership
        solver:         Solver type: 'Euler' or 'runge_kutta'



    """

    def __init__(self,groups,epsilon,beta,solver='Euler'):
        self.groups=groups
        self.epsilon=epsilon
        self.beta=beta
        self.solver=solver




    def _rho(self,N_list,C_list):

        #dimension
        D=len(N_list)

        #empty rho matrix
        rho=np.zeros(shape=(D,D))

        sigma_CN=np.sum(np.dot(N_list,C_list))

        for i in range(D):
            for j in range(D):
                rho[i,j]=(1-self.epsilon)*N_list[j]*C_list[j]/sigma_CN
                if i==j:
                    rho[i, j]+=self.epsilon

        return rho

    def _lambda(self,N_list,Y_inf_list,C_list):

        rho=self._rho(N_list, C_list)

        # dimension
        D = len(N_list)

        lambda_=[]


        for i in range(D):
            lambda_.append(C_list[i]*self.beta*np.sum([Y_inf_list[i] / N_list[i] for i in range(D)]))


        return lambda_










    def _states_dot(self,states,t):

        self.num_groups = len(self.groups)
        self.num_states = int(len(states) / self.num_groups)


        states_per_group=[states[i*self.num_states:(i+1)*self.num_states] for i in range(self.num_groups)]
        N_list=[np.sum(states_per_group[i][:]) for i in range(self.num_groups)]
        Y_inf_list=np.zeros(shape=self.num_groups)
        for i in range(self.num_groups):
            for j in [2,3,5]:
                Y_inf_list[i]+= states_per_group[i][j]
        C_list = [self.groups[i].C for i in range(self.num_groups)]



        _lambda=self._lambda(N_list,Y_inf_list,C_list)

        states_dot=[]

        for i in range(self.num_groups):
            states_dot.extend(self.groups[i].states_dot(states_per_group[i],_lambda[i]))

        return np.array(states_dot)


    def solve(self,total_time,time_step):

        self.total_time = total_time
        self.time_step = time_step


        initial_states=[]
        num_groups=len(self.groups)
        num_states=len(self.groups[0].initial_states)


        # time
        self.time = np.linspace(0, self.total_time, int(1 / (self.time_step)))


        for i in range(num_groups):
            initial_states.extend(self.groups[i].initial_states)


        if self.solver=='Euler':
            result=Euler(self._states_dot,initial_states,self.time)
        elif self.solver=='runge_kutta':
            result = odeint(self._states_dot, initial_states, self.time)


        result_per_group=np.zeros(shape=(num_states,len(self.time)))

        for i in range(num_groups):
            for j in range(num_states):
                for k in range(len(result)):
                    result_per_group[j][k]=result[k][i*num_states+j]
            self.groups[i].set_states(result_per_group,self.time)

    def bar_plot_incidence(self):
        values=np.zeros(shape=6)
        for i in range(len(self.groups)):
            values += np.array([self.groups[i].YP[-1], self.groups[i].YS[-1], self.groups[i].EL[-1], self.groups[i].YRS[-1], self.groups[i].L[-1], self.groups[i].Yt[-1]])
        labels = ['YP', 'YS', 'EL', 'YRS', 'L', 'Yt']
        plt.bar(labels, values)
        plt.ylabel('syphilis incidence per 100000')
        plt.show()

# Implicit Euler solver
def Euler(f, y0, time):
    y_i=y0
    y=[y_i]
    for i in range(len(time)-1):
        h = time[i+1] - time[i]
        y_i = opt.fsolve(lambda x: x-y_i-h*f(x, time[i]) ,y_i)
        y.append(y_i)
    return y





def main():

    #define constants
    miu = 0.033
    nu = 0.25
    tau_p = 0
    tau_s = 0
    tau_el = 0
    tau_rs = 0
    tau_L = 0
    tau_t = 0
    gamma = 0.2
    sigma1 = 1/21*365
    sigma2 = 1/46*365
    sigma3 = 1/108*365
    sigma4 = 0.5
    sigma5 = 0.033


    # C1 = 0.192
    # C2 = 2
    # C3 = 8.2
    #
    # beta=0.2

    # C1 = 0.864
    # C2 = 1.384
    # C3 = 7.005
    #
    # beta=0.2233

    C1 = 0.8642
    C2 = 1.3837
    C3 = 7.003

    beta = 0.22335

    tau=0.248









    g1=Syphilis_dynamics(name='Low',
                      miu=miu,
                      nu=nu,
                      tau_p=tau_p,
                      tau_s=tau_s,
                      tau_el=tau_el,
                      tau_rs=tau_rs,
                      tau_L=tau_L,
                      tau_t=tau_t,
                      gamma=gamma,
                      sigma1=sigma1,
                      sigma2=sigma2,
                      sigma3=sigma3,
                      sigma4=sigma4,
                      sigma5=sigma5,
                      C=C1)

    g2 = Syphilis_dynamics(name='Intermediate',
                           miu=miu,
                           nu=nu,
                           tau_p=tau_p,
                           tau_s=tau_s,
                           tau_el=tau_el,
                           tau_rs=tau_rs,
                           tau_L=tau_L,
                           tau_t=tau_t,
                           gamma=gamma,
                           sigma1=sigma1,
                           sigma2=sigma2,
                           sigma3=sigma3,
                           sigma4=sigma4,
                           sigma5=sigma5,
                           C=C2)

    g3 = Syphilis_dynamics(name='High',
                           miu=miu,
                           nu=nu,
                           tau_p=tau_p,
                           tau_s=tau_s,
                           tau_el=tau_el,
                           tau_rs=tau_rs,
                           tau_L=tau_L,
                           tau_t=tau_t,
                           gamma=gamma,
                           sigma1=sigma1,
                           sigma2=sigma2,
                           sigma3=sigma3,
                           sigma4=sigma4,
                           sigma5=sigma5,
                           C=C3)



    g1.set_initial_states(X0=71999,
                          E0=0.0,
                          YP0=1,
                          YS0=0.0,
                          EL0=0.0,
                          YRS0=0.0,
                          L0=0.0,
                          Yt0=0.0,
                          Z0=0.0)


    g2.set_initial_states(X0=14999,
                          E0=0.0,
                          YP0=1,
                          YS0=0.0,
                          EL0=0.0,
                          YRS0=0.0,
                          L0=0.0,
                          Yt0=0.0,
                          Z0=0.0)

    g3.set_initial_states(X0=12999,
                          E0=0.0,
                          YP0=1,
                          YS0=0.0,
                          EL0=0.0,
                          YRS0=0.0,
                          L0=0.0,
                          Yt0=0.0,
                          Z0=0.0)

    sim=Syphilis_simulation(groups=[g1,g2,g3],
                            epsilon=0,
                            beta=beta,
                            solver = 'Euler')
    sim.solve(total_time=10000,time_step=0.01)


    # g1.bar_plot_incidence()
    # g2.bar_plot_incidence()
    # g3.bar_plot_incidence()
    sim.bar_plot_incidence()



    g1.apply_screening_strategy(tau_p=tau,
                                 tau_s=tau,
                                 tau_el=tau,
                                 tau_rs=tau,
                                 tau_L=tau,
                                 tau_t=tau)
    g2.apply_screening_strategy(tau_p=tau,
                                tau_s=tau,
                                tau_el=tau,
                                tau_rs=tau,
                                tau_L=tau,
                                tau_t=tau)
    g3.apply_screening_strategy(tau_p=tau,
                                tau_s=tau,
                                tau_el=tau,
                                tau_rs=tau,
                                tau_L=tau,
                                tau_t=tau)

    incidence = (np.array(g1.incidence) + np.array(g2.incidence) + np.array(g3.incidence))
    print(incidence[-1])
    time = g1.time
    plt.plot(time, incidence, 'g', label='Incidence')
    plt.ylabel('syphilis incidence per 100000')
    plt.grid(color='b', linestyle='-')
    plt.show()


    sim.solve(total_time=10,time_step=0.01)

    sim.bar_plot_incidence()


    #post processing
    g1=sim.groups[0]
    g2=sim.groups[1]
    g3=sim.groups[2]

    incidence=(np.array(g1.incidence)+np.array(g2.incidence)+np.array(g3.incidence))
    print(incidence[-1])

    time=g1.time


    plt.plot(time, incidence, 'g', label='Incidence')
    plt.ylabel('syphilis incidence per 100000')
    plt.grid(color='b', linestyle='-')
    plt.show()




if __name__=='__main__':
    main()





