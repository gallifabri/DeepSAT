import environment
import bots.agent as standard
import bots.random_agent as rand
import bots.multi_layer_agent as multi
import bots.greedy as greedy
import bots.test_agent as test
from parser import parse_file
import numpy as np

np.set_printoptions(precision=2)

p1 = "problems/random30.dimacs"
p2 = "problems/random40.dimacs"
p3 = "problems/105.cnf"
p4 = "problems/sat5.dimacs"
p5 = "problems/random10.dimacs"
p6 = "problems/random20.dimacs"
p7 = "problems/demo.cnf"

episodes = 10000
bot = standard
problem = p1


variables, clauses = parse_file(problem)
env = environment.environment(variables,clauses)
agent = bot.agent(env, LEARNING_RATE = 0.0025, GAMMA = 0.99,  random_weights = True)



print("LOG: ***************************")
print("LOG: Number of variables: ", variables)
print("LOG: Number of clauses:\t", len(clauses))
print("LOG: Training with agent: ", agent.name)
print("LOG: ***************************")
print("LOG:")
print("LOG: ***************************")
print("LOG: ***** TRAINING AGENT ******")
print("LOG: ***************************")


sat, passes = agent.train(episodes,True)


print("LOG: ***************************")
print("LOG: *** TRAINING COMPLETED ****")
print("LOG: ***************************")
print("LOG:")
print("LOG: ***************************")
print("LOG: SATISFIABLE:\t", sat)
print("LOG: Number of passes:\t", passes)
print("LOG: ***************************")


