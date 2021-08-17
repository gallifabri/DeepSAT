import numpy as np
import matplotlib.pyplot as plt
import copy
import  environment
import statistics

#Hyperparameters
NUM_EPISODES = 10000
LEARNING_RATE = 0.00025
GAMMA = 0.99




class agent():
	env = None
	n = 0
	name = "Random Agent"
	alive = []
	num_var = 0

	def __init__(self, env, LEARNING_RATE = 0.0025, GAMMA = 0.99, random_weights = False):
		self.n = env.get_action_space_size()
		self.env  = env
		self.num_var = int(env.get_action_space_size() / 2)
		self.init_alive()
	
	def init_alive(self):
		self.alive = [i for i in range(0,self.num_var)]




	def reset(self):
		self.env.reset()

	def random_choice(self):


		index = np.random.choice(len(self.alive))
		var = self.alive[index]
		self.alive.remove(var)

		val = np.random.choice(2, p=[0.5,0.5])
		action = var*2 + val

		

		return action, var
	

	def train(self, episodes, agent):
		sat = False
		scores = []

		for e in range(0,episodes):
			sat, step = self.run_episode()
			scores.append(step)
			self.init_alive()


			if ((e+1) % 200 == 0):	
				print("LOG: Rollouts completed: ", (e+1), "Guessed: ", statistics.mean(scores[-200:]))
			
			if sat:
				break

		return sat, len(scores)

	def reinforce(self):

		sat = False	
		step = 0

		while True:
				step += 1

				# Environment chooses action to take
				action, label = self.random_choice()


				# agent performs action on env, gets observations back from env
				reward, consistent, sat = self.env.step(action)


				if not consistent:
					break

				if sat:
					break

		self.init_alive()

		return sat, step
