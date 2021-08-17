import numpy as np
import matplotlib.pyplot as plt
import copy
import  environment
import statistics


class agent():

	env = None
	w =  None
	name = "Standard Agent"

	#Hyperparameters
	NUM_EPISODES = 10000
	LEARNING_RATE = 0.0025
	GAMMA = 0.99
	random_weights = False


	def __init__(self, environment, LEARNING_RATE = 0.0025, GAMMA = 0.99, random_weights = False):
		self.env  = environment
		self.LEARNING_RATE = LEARNING_RATE
		self.GAMMA = GAMMA
		self.random_weights = random_weights

		self.initialize_weights()
		

	def reset(self):
		self.initialize_weights()
		self.env.reset()


	def initialize_weights(self):
		n = self.env.get_action_space_size()

		if self.random_weights:
			self.w = np.random.uniform(-1,1,(n + 1,n))
		else:
			self.w = np.full((n + 1, n), 0.5)


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


	def gradient(self, y, state, action):
	# computes the gradient of the policy

		dsoftmax = self.softmax_grad(y)[action,:]
		return state[None,:].T.dot(dsoftmax[None,:])


	def get_legal_actions(self):
	# Auxiliary method for choice()

		state = self.env.get_state()
		actions = []

		for i in range(0,len(state),2):
			legal = 1 - (state[i] + state[i+1])
			actions.append(legal)
			actions.append(legal)

		return actions


	def choice(self, y):
	# Creates a new probability distribution excluding the illegal actions
	# Returns random choice of action (given the distribution), and name of variable

		legal_actions = self.get_legal_actions()
		legal_actions_probs = np.multiply(legal_actions, y)
				
		average_difference = (sum(y) - sum(legal_actions_probs)) / sum(legal_actions)
		probs = [((legal_actions_probs[i] + average_difference)*legal_actions[i]) for i in range(0,len(legal_actions))]

		return np.random.choice(self.env.get_action_space_size(), p=probs)


	def train(self, episodes, verbose):
	# Executes rollouts until solution is found or until max_num of episodes reached

		sat = False
		scores = []

		for e in range(0,episodes):
			sat, step = self.reinforce()
			scores.append(step)


			if ((e+1) % 100 == 0):	
				print("LOG: Rollouts completed: ", (e+1), "Guessed: ", statistics.mean(scores[-100:]))
			
			if sat:
				break

		return sat, len(scores)



	def reinforce(self):
		grads = []	
		rewards = []

		while True:
			state = self.env.get_state_with_bias_node()

			# FORWARD PROPAGATION
			probs = self.policy(state,self.w)

			# choose action to take
			action = self.choice(probs)

			# agent performs action on environment, gets observations back
			reward, consistent, sat = self.env.step(action)
			
			# COMPUTING THE GRADIENT
			grad = self.gradient(probs,state, action)
			log_grad = grad / probs[action]

			# store data for parameters update
			grads.append(log_grad)
			rewards.append(reward)	



			if not consistent:
				break

			if sat:
				return True, reward


		for i in range(len(grads)):
			# Update the weights given by the REINFORCE formula
			# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
			self.w += self.LEARNING_RATE * grads[i] * sum([ r * (self.GAMMA ** t) for t,r in enumerate(rewards[i:])])


		if len(rewards) == 1:
			r = rewards[0]
		else:
			r = rewards[-2]

		return False, r

