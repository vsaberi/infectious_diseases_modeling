import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""""
Author: Vahid Saberi
Date: Feb 25, 2018

Description:

The code regenerates the simulations of the following paper:

Mishra, Sharmistha, David N. Fisman, and Marie-Claude Boily.
"The ABC of terms used in mathematical models of infectious diseases."
Journal of Epidemiology & Community Health (2010): jech-2009.
""""



def model_A(X,t):
    S=X[0]
    I=X[1]

    #parameters
    R0=21
    r_R=1/70

    N=S+I

    lambda_ = R0 * r_R * I / N

    X_dot=[]
    X_dot.append(r_R*N-lambda_*S-r_R*S)
    X_dot.append(lambda_*S-r_R*I)

    return X_dot

def model_B(X,t):
    S=X[0]
    I=X[1]

    #parameters
    R0=3
    r_R=0
    r_L=0.1

    X_dot=[]
    N=S+I
    lambda_=R0*(r_R+r_L)*I/N
    X_dot.append(r_R*N-lambda_*S-r_R*S+r_L*I)
    X_dot.append(lambda_*S-r_R*I-r_L*I)

    return X_dot




def model_C(X,t):
    S=X[0]
    I=X[1]
    R=X[2]

    #parameters
    R0=3
    r_R=0
    r_l=0.1

    X_dot=[]
    N=S+I+R
    lambda_=R0*(r_R+r_l)*I/N

    X_dot.append(r_R*N-lambda_*S-r_R*S)
    X_dot.append(lambda_*S-r_R*I-r_l*I)
    X_dot.append(r_l*I-r_R*R)


    return X_dot



def model_D(X,t):
    S=X[0]
    I=X[1]
    R=X[2]

    #parameters
    R0=3
    r_R=0
    r_l=0.1
    r_L=0.006

    X_dot=[]
    N=S+I+R
    lambda_=R0*(r_R+r_l)*I/N

    X_dot.append(r_R*N-lambda_*S+r_L*R-r_R*S)
    X_dot.append(lambda_*S-r_R*I-r_l*I)
    X_dot.append(r_l*I-r_L*R-r_R*R)


    return X_dot




#Solve the models

def solve_model_A():

    # initial condition
    X0 =[1-0.001,0.001]

    # time points
    t = np.linspace(0, 600,10000)

    # solve ODE
    X = odeint(model_A, X0, t)

    S=[X[i][0] for i in range(len(X))]
    I=[X[i][1] for i in range(len(X))]


    # plot results
    plt.plot(t,S,'b',label='susceptible')
    plt.plot(t,I,'r',label='infectious')
    plt.xlabel('time (days)')
    plt.ylabel('Population fraction')
    plt.legend()
    plt.title('Model (A)')
    plt.show()


def solve_model_B():

    # initial condition
    X0 =[1-0.001,0.001]

    # time points
    t = np.linspace(0, 600,10000)

    # solve ODE
    X = odeint(model_B, X0, t)

    S=[X[i][0] for i in range(len(X))]
    I=[X[i][1] for i in range(len(X))]


    # plot results
    plt.plot(t,S,'b',label='susceptible')
    plt.plot(t,I,'r',label='infectious')
    plt.xlabel('time (days)')
    plt.ylabel('Population fraction')
    plt.legend()
    plt.title('Model (B)')
    plt.show()


def solve_model_C():

    # initial condition
    X0 =[1-0.001,0.001,0]

    # time points
    t = np.linspace(0, 600,10000)

    # solve ODE
    X = odeint(model_C, X0, t)

    S=[X[i][0] for i in range(len(X))]
    I=[X[i][1] for i in range(len(X))]
    R=[X[i][2] for i in range(len(X))]


    # plot results
    plt.plot(t,S,'b',label='susceptible')
    plt.plot(t,I,'r',label='infectious')
    plt.plot(t,R,'g',label='recovered')
    plt.xlabel('time (days)')
    plt.ylabel('Population fraction')
    plt.legend()
    plt.title('Model (C)')
    plt.show()



def solve_model_D():

    # initial condition
    X0 =[1-0.001,0.001,0]

    # time points
    t = np.linspace(0, 600,10000)

    # solve ODE
    X = odeint(model_D, X0, t)

    S=[X[i][0] for i in range(len(X))]
    I=[X[i][1] for i in range(len(X))]
    R=[X[i][2] for i in range(len(X))]


    # plot results
    plt.plot(t,S,'b',label='susceptible')
    plt.plot(t,I,'r',label='infectious')
    plt.plot(t,R,'g',label='recovered')
    plt.xlabel('time (days)')
    plt.ylabel('Population fraction')
    plt.legend()
    plt.title('Model (D)')
    plt.show()


if __name__=="__main__":
    solve_model_D()