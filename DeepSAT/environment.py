import numpy as np
import collections


class environment():
	
	working_CNF =  []
	input_CNF = []
	
	state = []
	instances_V  = {}


	def __init__(self, num_of_var, CNF):
		self.state = np.zeros(2 * num_of_var)
		self.input_CNF = CNF
		self.working_CNF = [c.copy() for c in self.input_CNF]
			

	def get_state_with_bias_node(self): 
		return np.append([1], self.state)

	def get_number_of_variables(self):
		return int(len(self.get_state())/2)


	def get_action_space_size(self):
		return len(self.state)


	def get_state(self):
		return self.state


	def reward(self, consistent):
		if not consistent:
			return 0
		return sum(self.state)


	def consistent(self, instances):
	# Returns consistent = False if a contradiction has been found in the clauses
	# Retruns consistent = True otherwise
	# Returns sat = True if CNF is satisfiable

		sat =  False
		CNF = self.working_CNF

		for c in CNF:
			size = len(c)

			for literal in c:
				key = abs(literal)
				value = literal > 0

				if key in instances:
					if value == instances[key]:
						CNF.remove(c)
						break
					else:
						size -= 1
				else:
					break
			
			if size == 0:
				return False, sat


		if (len(instances) == self.get_number_of_variables()):
			sat = True

		return True, sat


	def step(self, action):

		self.state[action] =  1
		self.instances_V[int(action/2)-1] = (action % 2 == 0)		
		
		consistent, sat = self.consistent(self.instances_V)
		reward = self.reward(consistent)

		if not consistent:
			self.reset()

		return reward, consistent, sat


	def reset(self):
		self.state = np.dot(self.state, 0)
		self.instances_V = {}
		self.working_CNF = self.working_CNF = [c.copy() for c in self.input_CNF]

