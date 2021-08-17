import numpy as np
import matplotlib.pyplot as plt
import copy
import  environment
import statistics


class agent():
	env = None
	w =  None
	name = "Standard Agent"
	random_param = False
	n = 0

	#Hyperparameters
	NUM_EPISODES = 10000
	LEARNING_RATE = 0.0025
	GAMMA = 0.99

	def __init__(self, env, LEARNING_RATE = 0.0025, GAMMA = 0.99, random_weights = False):
		n = env.get_action_space_size()
		self.n = n
		self.env  = env
		
		self.LEARNING_RATE = LEARNING_RATE
		self.GAMMA = GAMMA
		self.random_param = random_weights

		if random_param:
			self.w = np.random.uniform(-1,1,(n + 1,n))
		else:
			self.w = np.full((n + 1, n), 0.5)


	def reset(self):
		if self.random_param:
			self.w = np.random.uniform(-1,1,(self.n + 1,self.n))
		else:
			self.w = np.full((self.n + 1, self.n), 0.5)

		self.env.reset()


	def policy(self,state,w):
	# Performs the softmax forwards propagation
	# Gives as output a probability distribution 

		z = state.dot(w)
		exp = np.exp(z)
		return exp/np.sum(exp)

	
	def softmax_grad(self,softmax):
	# Vectorized softmax Jacobian

	    s = softmax.reshape(-1,1)
	    return np.diagflat(s) - np.dot(s, s.T)

	def choice2(self,state,y):
		maxi =  abs(y[0]  - y[1]) + 1
		index = 0

		for i in range(0,len(y),2):
			if state[i+1] == 1 or state[i+2] == 1:
				continue

			dif = abs(y[i]  - y[i+1])

			if dif <= maxi:
				maxi = dif
				index = i

		pair = [y[index], y[index+1]]

		exp = np.exp(pair)
		soft =  exp/np.sum(exp)

		pre_action = np.random.choice(2, p=soft)
		action = index + pre_action

		label = int(action / 2) 

		return action, label

	def train(self, episodes, agent, verbose):
		sat = False
		scores = []

		for e in range(0,episodes):
			sat, step = self.run_episode()
			scores.append(step)

			if ((e+1) % 200 == 0):	
				if verbose:
					print("LOG: Rollouts completed: ", (e+1), "Guessed: ", statistics.mean(scores[-200:]))
			
			if sat:
				break

		return sat, len(scores)

	def run_episode(self):

		sat = False
		state = self.env.get_state_with_bias_node()

		grads = []	
		rewards = []

		# Keep track of game score to print
		score = 0
		step = 0

		while True:
				step += 1

				# FORWARD PROPAGATION
				probs = self.policy(state,self.w)

				# Environment chooses action to take
				action, label = self.choice2(state, probs)
				# print()
				# print(probs)
				# print(action)
				# print()

				# agent performs action on env, gets observations back from env
				next_state, reward, consistent, sat = self.env.step(action,label)

				# COMPUTING THE GRADIENT
				dsoftmax = self.softmax_grad(probs)[action,:]
				dlog = dsoftmax / probs[action]
				grad = state[None,:].T.dot(dlog[None,:])

				counter = action - 1

				if action % 2 == 0:
					counter = action + 1



				# Save values for updating the parameters
				grads.append(grad)
				rewards.append(reward)		

				score+=reward

				state = next_state
				if not consistent:
					break

				if sat:
					break



		#factor =  abs(step - mean) / mean
		#print(factor)

		for i in range(len(grads)):
			#print(LEARNING_RATE)
			# print(self.w)
			# # print()
			# # print(self.w.sum(axis=0))
			# # print()
			# print(grads[i])


			# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
			self.w += self.LEARNING_RATE * grads[i] * sum([ r * (self.GAMMA ** t) for t,r in enumerate(rewards[i:])])
			continue

		return sat, step

