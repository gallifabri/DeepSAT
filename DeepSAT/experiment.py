import environment
import bots.agent as standard
import bots.greedy as greedy
import bots.random_agent as rand
import bots.multi_layer_agent as multi
from parser import parse_file
import numpy as np
import statistics
from tqdm import tqdm

#### 1. Choose problem to solve ####

problem1 = "problems/random30.dimacs"
problem2 = "problems/random40.dimacs"
problem3 = "problems/105.cnf"
problem4 = "problems/random20.dimacs"


path_problem = problem2

#### 2. Chose hyperparameters ####

NUM_EPISODES = 100000
LEARNING_RATE = 0.0025
GAMMA = 0.99
random_weights = False


#### 3. Choose bot ####

bot = rand


#### 4. Choose save options ####

save = True
path_name = "experiment_data/"
file_name = "findings19"


#### 5. Choose type of experiment ####

exp_type = 2
repeats = 20


#### 6. Add comments ####

comment = ""

#######################


variables, clauses = parse_file(path_problem)

env = environment.environment(variables,clauses)
agent = bot.agent(env, LEARNING_RATE, GAMMA, random_weights = random_weights)







def train_detail(self, episodes):
		sat = False
		num_steps = []
		means = []
		rollouts = 0

		print("\nCOLLECTING DATA.")

		for e in tqdm(range(0,episodes)):
			rollouts += 1
			sat, step = self.reinforce()
		
			num_steps.append(step)
			means.append(np.mean(num_steps[-20:]))
			
			if sat:
				print("Solution Found.")
				break


		return rollouts, sat, num_steps, means




def train_simple(self, episodes, repeats):
		num_rollouts = []
		results = []

		print("\nCOLLECTING DATA.")
		
		for r in tqdm(range(0,repeats)):

			sat = False
			rollouts = 0

			for e in range(0,episodes):
				rollouts += 1
				sat, step = self.reinforce()
			
				
				if sat:
					break

			num_rollouts.append(rollouts)
			results.append(sat)
			rollouts = 0
			self.reset()

		return num_rollouts, results, np.mean(num_rollouts)




def stringify1(rollouts,sat):
	s = "###### EXPERIMENT TYPE 1 : " + file_name + "######\n\n"

	s += "Agent: " + agent.name + "\n\n"

	s += "Problem:  " +  path_problem  + "\n"
	s += "Variables: " + str(variables) + "\n"
	#s += "Clauses: " + str(len(env.CNF)) + "\n\n"

	s += "HYPER PARAMETERS: \n\n"
	s += "LEARNING_RATE = " + str(LEARNING_RATE) + "\n"
	s += "GAMMA = " + str(GAMMA) + "\n"
	s += "MAX_EPISODES = " + str(NUM_EPISODES) + "\n"
	s += "random_weights = " + str(random_weights) + "\n"
	s += "\n\n"

	s += "Results: \n"
	s += "Solution found: " + str(sat) + "\n"
	s += "rollouts: " + str(rollouts) +  "\n\n"

	s += "Comment: " + comment + "\n\n"

	s += "###### END ######"

	return s




def stringify2(repeats,mean,dev,solved):
	s = "###### EXPERIMENT TYPE 2 : " + file_name + "######\n\n"

	s += "Agent: " + agent.name + "\n\n"

	s += "Problem:  " +  path_problem  + "\n"
	s += "Variables: " + str(variables) + "\n"
	#s += "Clauses: " + str(len(env.CNF)) + "\n\n"

	s += "HYPER PARAMETERS: \n\n"
	s += "LEARNING_RATE = " + str(LEARNING_RATE) + "\n"
	s += "GAMMA = " + str(GAMMA) + "\n"
	s += "MAX_EPISODES = " + str(NUM_EPISODES) + "\n"
	s += "random_weights = " + str(random_weights) + "\n"
	s += "\n\n"

	s += "Results: \n"
	s += "Repeats: " + str(repeats) + "\n"
	s += "Mean: " + str(mean) +  "\n"
	s += "Standard Dev.: " + str(dev) + "\n"
	s += "Solved : " + str(solved) + "%\n\n"

	s += "Comment: " + comment + "\n\n"

	s += "###### END ######"

	return s







if exp_type == 1:
	data = []
	rollouts, sat, num_steps, means = train_detail(agent,NUM_EPISODES)
	data.append([rollouts, sat, num_steps, means])

	if save:
		np.save(path_name + file_name + "_numsteps", num_steps)
		np.save(path_name + file_name + "_means", means)
		s = stringify1(rollouts,sat)

		f = open(path_name + file_name + "_detail" + ".txt", "w")
		f.write(s)
		f.close()


else:
	num_rollouts, results, mean = train_simple(agent,NUM_EPISODES,repeats)

	solved = sum([1 for res in results if res]) / len(results) * 100

	if save:
		np.save(path_name + file_name + "_numrollouts", num_rollouts)
		np.save(path_name + file_name + "_results", results)
		
		s = stringify2(repeats,mean, np.std(num_rollouts), solved)

		f = open(path_name + file_name + "_detail" + ".txt", "w")
		f.write(s)
		f.close()

