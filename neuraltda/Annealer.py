import .spectralAnalysis as sa 
import .simpComp as sc 
import numpy as np 
from joblib import Parallel, delayed

class Annealer:

	def __init__(self, loss, system, eps, K, beta=0.15):
		self.loss = loss
		self.system = system
		# loss is an object that has an "error" method
		# system is an object that has an initialize, run method
		self.s = system.initialize()
		#self.out = system.run(self.s)
		self.E = np.inf 
		self.T = np.inf
		self.eps = eps
		self.K = K
		self.beta = beta
		self.n_trials_at_temp = 5
		self.dt = 0.99

	def anneal(self, kmax):
		for k in range(kmax):
			self.T = self.dt**k 
			for t in range(self.n_trials_at_temp):
				s_new = self.neighbor(self.s)
				out_new = self.system.run(s_new)
				#print(np.any((out_new - self.out) != 0 ))
				E_new = self.loss.loss(out_new, self.beta)
				print('Status: {}/{}, E: {}, E_new: {}, Temp: {}'.format(k, kmax, self.E, E_new, self.T))
				if self.accept_prob(E_new) >= np.random.rand():

					self.s = s_new
					self.out = out_new
					self.E = E_new
			
		return self.s

	def accept_prob(self, E_new):
		if E_new < self.E:
			return 1.0
		else:
			return np.exp(-(E_new - self.E) / self.T)

	def neighbor(self, s_old):
		ds = np.random.standard_normal(np.shape(s_old))
		ds = (self.eps*(np.random.rand()+0.5))*(ds / np.sum(np.power(ds, 2)))
		return s_old + ds

	def temperature(self, k, kmax):

		return self.K*np.exp(-1.0/(1.0 - np.float(k)/np.float(kmax)))