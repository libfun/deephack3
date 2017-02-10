import numpy as np

class Player(object):

	def __init__(self, env, agents):
		self.env = env
		self.agents = {agent.name: agent for agent in agents}
		self.best_agent = None

	def choose_and_record(self, num_episodes, self.env):

		scores = {}

		for agent in self.agents.keys:
			scores[agent] = self.agents[agent].play(num_episodes, record = False)

		self.agents[max(scores, key=scores.get)].play(num_episodes, record = True)





