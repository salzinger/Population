import scipy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import sdeint

'''
Overview:
The SEIRS models the flows of people between four states: susceptible (S), exposed (E), infected (I), and resistant (R).
Each of those variables represents the number of people in those groups.
The parameters alpha and beta partially control how fast people move from being susceptible to exposed (beta), from exposed to infected (sigma),
 and from infected to resistant (gamma). This model has two additional parameters; one is the background mortality (mu) which is unaffected by disease-state,
  while the other is vaccination (nu). The vaccination moves people from the susceptible to resistant directly, without becoming exposed or infected.
  
The SEIRS differs from the SEIR model by letting recovered individuals lose their resistance over time. The rate at which people lose their infection is governed by the parameter rho.

Beta	The parameter controlling how often a susceptible-infected contact results in a new exposure.
Gamma	The rate an infected recovers and moves into the resistant phase.
Sigma	The rate at which an exposed person becomes infective.
Mu	The natural mortality rate (this is unrelated to disease). This models a population of a constant size,
Nu	The rate at which susceptible become vaccinated.
Rho	The rate at which resistant people lose their resistance and become susceptible again.
Initial susceptible	The number of susceptible individuals at the beginning of the model run.
Initial exposed	The number of exposed individuals at the beginning of the model run.
Initial infected	The number of infected individuals at the beginning of the model run.
Initial recovered	The number of recovered individuals at the beginning of the model run.
Days	Controls how long the model will run.
'''

beta, gamma, sigma, theta = (2.6, 1/20, 1/20, 0.01)

mu, nu, rho = (0, 0, 0)

susceptible, exposed, infected, recovered, dead = (10**6, 0, 150, 0, 0)

N = susceptible + exposed + infected + recovered + dead

def seirs(t, z):
    S, E, I, R, D, N = z
    return [mu*(N-S) + rho*R - beta*S*I/N - nu*S,  #susceptible
            beta*S*I/N - (mu+sigma)*E,             #exposed
            sigma*E - (mu+gamma+theta)*I,          #infected
            gamma*I - mu*R + nu*S - rho*R,         #recovered
            theta*I,                               #dead
            0]

Gamma, Omega, Delta = (6, 40,0)

gg0, ee0, rr0, REge0, IMge0, REgr0, IMgr0, REer0, IMer0, trace0 = (0, 0, 1, 0, 0, 0, 0, 0, 0, 1)

def coherence(t, z):
    gg, ee, rr, REge, IMge, REgr, IMgr, REer, IMer, trace = z
    return [Gamma * ee,
            - Omega * IMer - Gamma * ee,
            Omega * IMer,
            - Gamma/2 * (REge + IMge),
            Omega/2 * (REgr + IMgr) - Delta * (REge + IMge),
            0,
            Omega/2 * (REge + IMge) - Delta * (REgr + IMgr),
            - Gamma/2 * (REge + IMer),
            Omega/2 * (ee - rr) - Delta * (REer + IMer),
            0]


#sol = solve_ivp(seirs, [0, 120], [susceptible, exposed, infected, recovered, dead, N], dense_output=True)
sol = solve_ivp(coherence, [0, 1], [gg0, ee0, rr0, REge0, IMge0, REgr0, IMgr0, REer0, IMer0, trace0], dense_output=True)
#print(sol)

def opener(i):
    data=[]
    file = open('dat.txt')
    country = file.readline().split()[i+1]
    file.close()
    file = open('newdat.txt')
    for line in file:
        try:
            data.append(float(line.split()[i+1]))
        except:
            None
    file.close()
    return country,data

t=0
print(opener(1))


'''
Spain=[line.split()[2] for line in file]
France=[line.split()[3] for line in file]
Germany=[line[0:2] for line in file]
US=[line[0:2] for line in file]
UK=[line[0:2] for line in file]
Belgium=[line[0:2] for line in file]
Netherlands=[line[0:2] for line in file]
Norway=[line[0:2] for line in file]
Sweden=[line[0:2] for line in file]
Switzerland=[line[0:2] for line in file]
Austria=[line[0:2] for line in file]
Australia=[line[0:2] for line in file]
Canada=[line[0:2] for line in file]
Israel=[line[0:2] for line in file]
Brazil=[line[0:2] for line in file]
Portugal=[line[0:2] for line in file]
Turkey=[line[0:2] for line in file]
'''




fig, ax = plt.subplots(3,1)
#plt.yscale('log')
ax[0].plot(sol.t, sol.y[0], label="gg", marker='', linestyle='-')
ax[0].plot(sol.t, sol.y[1], label="ee", marker='', linestyle='-',color='orange')
ax[0].plot(sol.t, sol.y[2], label="rr", marker='', linestyle='-',color='r')
#ax[0].plot(sol.t, sol.y[3], label="Recovered", marker='', linestyle='-',color='g')
#ax[0].plot(sol.t, sol.y[4], label="Dead", marker='', linestyle='--',color='grey')
#ax[0].plot(sol.t, sol.y[5], label="Total", marker='',color='grey')
#plt.ylim(1,10**10)
# ax.plot(times, result.expect[1],label="MagnetizationZ",linestyle='--',marker='o',markersize='2');
# ax.plot(times, result.expect[2],label="Exp(SigmaZ,0)");
# ax.plot(times, result.expect[3],label="Exp(SigmaX,0)",linestyle='--');
# ax.plot(times, np.abs(ups),label="Tr(rho_0,uu)",linestyle='--');
# ax.plot(times, np.abs(downs),label="Tr(rho_0,dd)",linestyle='-');
#ax[0].set_xlabel('Days')
#ax[0].set_ylabel('')
ax[0].legend(loc="upper right")


for i in range(0,24):
    try:
        None#ax[1].plot(range(0, len(opener(i)[1])), opener(i)[1], label=opener(i)[0], marker='o', linestyle='', markersize='2')
    except:
        None
#ax[1].plot(sol.t, sol.y[2]+sol.y[3]+sol.y[4], label="Total Infected", marker='', linestyle='-',color='r')
#plt.xlim(0,60)

ax[1].plot(sol.t, sol.y[3], label="Re[ge]", marker='', linestyle='-')
ax[1].plot(sol.t, sol.y[4], label="Im[ge]", marker='', linestyle='-',color='orange')
ax[1].plot(sol.t, sol.y[5], label="Re[gr]", marker='', linestyle='-',color='r')
ax[1].plot(sol.t, sol.y[6], label="Im[gr]", marker='', linestyle='-',color='g')
ax[1].plot(sol.t, sol.y[7], label="Re[er]", marker='', linestyle='--',color='grey')
ax[1].plot(sol.t, sol.y[8], label="Im[er]", marker='',color='grey')
# ax.plot(times, result.expect[1],label="MagnetizationZ",linestyle='--',marker='o',markersize='2');
# ax.plot(times, result.expect[2],label="Exp(SigmaZ,0)");
# ax.plot(times, result.expect[3],label="Exp(SigmaX,0)",linestyle='--');
# ax.plot(times, np.abs(ups),label="Tr(rho_0,uu)",linestyle='--');
# ax.plot(times, np.abs(downs),label="Tr(rho_0,dd)",linestyle='-');
#ax[1].set_xlabel('Days')
#ax[1].set_ylabel('')
ax[1].legend(loc="right")
#plt.show()

A = np.array([[-0.1, 0.5],
              [-0.5, 0.1]])

B = np.diag([0.5, 0.5]) # diagonal, so independent driving Wiener processes
B = np.array([[-0.0, -5],
              [0.0, 0]])
tspan = np.linspace(0.0, 10.0, 10001)
x0 = np.array([1.0, 0.1])

def f(x, t):
    return A.dot(x)

def G(x, t):
    return B

result = sdeint.stratint(f, G, x0, tspan)
print(result)
ax[2].plot(tspan, result)
plt.show()
