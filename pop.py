import scipy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

beta, gamma, sigma, theta = (3, 1/20, 0.5, 0.01/20)

mu, nu, rho = (0, 0, 0)

susceptible, exposed, infected, recovered, dead = (7*10**9, 1, 0, 0, 0)

N = susceptible + exposed + infected + recovered + dead

def seirs(t, z):
    S, E, I, R, D, N = z
    return [mu*(N-S) + rho*R - beta*S*I/N - nu*S,  #susceptible
            beta*S*I/N - (mu+sigma)*E,             #exposed
            sigma*E - (mu+gamma+theta)*I,          #infected
            gamma*I - mu*R + nu*S - rho*R,         #recovered
            theta*I,                               #dead
            0]


sol = solve_ivp(seirs, [0, 1800], [susceptible, exposed, infected, recovered, dead, N], dense_output=True)

print(sol)

fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0], label="Susceptible", marker='', linestyle='-')
ax.plot(sol.t, sol.y[1], label="Exposed", marker='', linestyle='-',color='orange')
ax.plot(sol.t, sol.y[2], label="Infected", marker='', linestyle='-',color='r')
ax.plot(sol.t, sol.y[3], label="Recovered", marker='', linestyle='-',color='g')
ax.plot(sol.t, sol.y[4], label="Dead", marker='', linestyle='--',color='grey')
ax.plot(sol.t, sol.y[5], label="Total", marker='',color='grey')
plt.yscale('log')
plt.ylim(1,10**10)
# ax.plot(times, result.expect[1],label="MagnetizationZ",linestyle='--',marker='o',markersize='2');
# ax.plot(times, result.expect[2],label="Exp(SigmaZ,0)");
# ax.plot(times, result.expect[3],label="Exp(SigmaX,0)",linestyle='--');
# ax.plot(times, np.abs(ups),label="Tr(rho_0,uu)",linestyle='--');
# ax.plot(times, np.abs(downs),label="Tr(rho_0,dd)",linestyle='-');
ax.set_xlabel('Days')
ax.set_ylabel('')
leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()
