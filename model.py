
"""""
Author: Vahid Saberi
Date: Feb 25, 2018

Description:

The purpose of this code is to regenerates the simulations of the following paper:

Mishra, Sharmistha, David N. Fisman, and Marie-Claude Boily.
"The ABC of terms used in mathematical models of infectious diseases."
Journal of Epidemiology & Community Health (2010): jech-2009.

The includes a model of SIRS and SIS structures used for mathematical modeling
of infectious diseases

"""

#import required moduls

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt




class SIRS_model:

    """
    This class defines a SIRS model for infectious disease modeling.
    For more info see the above-mentioned paper.

    Model parameters:

    r_R:        Rate of death
    r_L:        Rate of loss of immunity
    r_l:        Rate of developing protective immunity
    R0:         reproduction ratio

    Model solver inputs:

    S0:         Initial population fraction of susceptible hosts
    I0:         initial population fraction of infectious people
    R0:         Initial population fraction of recovered people
    t:          Simulation duration
    dt:         Time step size


    Model outputs:

    t:         Time
    S(t):         Time series of population fraction of susceptible hosts
    I(t):         Time series of population fraction of infectious people
    R(t):         Time series of population fraction of recovered people

    """

    def __init__(self,r_R,r_L,r_l,R0):


        # parameters
        self.R0 = R0
        self.r_R = r_R
        self.r_l = r_l
        self.r_L = r_L


    def _model(self,X, t):

        """"
        This function is an internal function that models the dynmaics.
        It takes the X=[S,I,R] and time vector and returns X_dot.
        """

        S = X[0]
        I = X[1]
        R = X[2]

        X_dot = []
        N = S + I + R
        lambda_ = self.R0 * (self.r_R + self.r_l) * I / N

        X_dot.append(self.r_R * N - lambda_ * S + self.r_L * R - self.r_R * S)
        X_dot.append(lambda_ * S - self.r_R * I - self.r_l * I)
        X_dot.append(self.r_l * I - self.r_L * R - self.r_R * R)

        return X_dot

    def solve(self,S_0,I_0,R_0,t,dt):

        # initial condition
        X0 = [S_0, I_0, R_0]

        #time
        time = np.linspace(0, t, int(1/(dt)))

        # solve ODE
        X = odeint(self._model, X0, time)

        S = [X[i][0] for i in range(len(X))]
        I = [X[i][1] for i in range(len(X))]
        R = [X[i][2] for i in range(len(X))]

        return time,S,I,R




class SIS_model:

    """
    This class defines a SIS model for infectious disease modeling.
    For more info see the above-mentioned paper.

    Model parameters:

    r_R:        Rate of death
    r_L:        Rate of loss of immunity
    R0:         reproduction ratio

    Model solver inputs:

    S0:         Initial population fraction of susceptible hosts
    I0:         initial population fraction of infectious people
    t:          Simulation duration
    dt:         Time step size


    Model outputs:

    t:         Time
    S(t):         Time series of population fraction of susceptible hosts
    I(t):         Time series of population fraction of infectious people

    """

    def __init__(self,r_R,r_L,R0):


        # parameters
        self.R0 = R0
        self.r_R = r_R
        self.r_L = r_L


    def _model(self,X, t):

        """"
        This function is an internal function that models the dynmaics.
        It takes the X=[S,I] and time vector and returns X_dot.
        """

        S=X[0]
        I=X[1]


        X_dot = []
        N = S + I
        lambda_ = self.R0 * (self.r_R + self.r_L) * I / N

        X_dot.append(self.r_R * N - lambda_ * S + self.r_L * I - self.r_R * S)
        X_dot.append(lambda_ * S - self.r_R * I - self.r_L * I)

        return X_dot

    def solve(self,S_0,I_0,t,dt):

        # initial condition
        X0 = [S_0, I_0]

        #time
        time = np.linspace(0, t, int(1/(dt)))

        # solve ODE
        X = odeint(self._model, X0, time)

        S = [X[i][0] for i in range(len(X))]
        I = [X[i][1] for i in range(len(X))]

        return time,S,I



#plotting tool
def plot_SIR(time,S=None,I=None,R=None,title='Model (D)',x_label='time (days)',y_label='Population fraction'):
    # plot results
    if S is not None:
        plt.plot(time, S, 'b', label='susceptible')
    if I is not None:
        plt.plot(time, I, 'r', label='infectious')
    if R is not None:
        plt.plot(time, R, 'g', label='recovered')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.show()







#Solve the models corresponding to plots A, B, C and D in the paper

def solve_model_A():

    model_A=SIRS_model(r_R=1/70,r_L=0,r_l=0,R0=21)
    time,S,I,_=model_A.solve(S_0=0.999,I_0=0.001,R_0=0,t=600,dt=0.001)

    plot_SIR(time=time,S=S,I=I,R=None,title='Model (A)')


def solve_model_B():

    model_B=SIS_model(r_R=0,r_L=0.1,R0=3)
    time,S,I=model_B.solve(S_0=0.999,I_0=0.001,t=600,dt=0.001)

    plot_SIR(time=time,S=S,I=I,R=None,title='Model (B)')


def solve_model_C():

    model_C=SIRS_model(r_R=0,r_L=0,r_l=0.1,R0=3)
    time,S,I,R=model_C.solve(S_0=0.999,I_0=0.001,R_0=0,t=600,dt=0.001)

    plot_SIR(time=time,S=S,I=I,R=R,title='Model (C)')

def solve_model_D():

    model_D=SIRS_model(r_R=0,r_L=0.006,r_l=0.1,R0=3)
    time,S,I,R=model_D.solve(S_0=0.999,I_0=0.001,R_0=0,t=600,dt=0.001)

    plot_SIR(time=time,S=S,I=I,R=R,title='Model (D)')





if __name__=="__main__":
    # solve_model_A()
    # solve_model_B()
    # solve_model_C()
    solve_model_D()