import numpy as np
import matplotlib.pyplot as plt
import copy
import  environment
import statistics



class agent():

	#Hyperparameters
	NUM_EPISODES = 10000
	LEARNING_RATE = 0.00025
	GAMMA = 0.99

	random_param = False
	env = None
	w =  []
	n = 0
	name = "Multi-Layer Agent"

	def __init__(self, env, LEARNING_RATE = 0.0025, GAMMA = 0.99, random_weights = False):
		self.n = env.get_action_space_size()
		self.env  = env

		self.LEARNING_RATE = LEARNING_RATE
		self.GAMMA = GAMMA
		self.random_param = random_weights
		
		self.set_weights()


			


	def reset(self):
		self.set_weights()
		self.env.reset()

	def set_weights(self):
		n = self.n
		x = n
		
		for i in range(0, self.n, 2):
			if self.random_param:
				self.w.append(np.random.uniform(-1,1,(n + 1,x)))
			else:
				self.w.append(np.full((n + 1,x), 0.5))
				
			x -= 2

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


	def annonymous_choice(self, y):
		action = np.random.choice(len(y), p=y)

		label = self.env.alive[int(action/2)]

		return action, (label)



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
		W = None
		#print("RUNNING EPISODE!")
		while True:
				W = self.w[step]
				step += 1

				# FORWARD PROPAGATION
				probs = self.policy(state,W)

				# Environment chooses action to take
				action, label = self.annonymous_choice(probs)

				# agent performs action on env, gets observations back from env
				next_state, reward, consistent, sat = self.env.step(action, label)

				# COMPUTING THE GRADIENT
				dsoftmax = self.softmax_grad(probs)[action,:]
				dlog = dsoftmax / probs[action]
				grad = state[None,:].T.dot(dlog[None,:])

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

		alfa = self.LEARNING_RATE
		#print(alfa)

		for i in range(len(grads)):
			W = self.w[i]
			num = self.n - i
			beta = alfa/(num)
			#beta = alfa * (GAMMA ** num) 
			#beta = max(beta, 0.025)
			#beta = min(beta, 0.05)


			# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
			W += beta * grads[i] * sum([ r * (self.GAMMA ** t) for t,r in enumerate(rewards[i:])])
			continue

		return sat, step
