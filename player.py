import numpy as np
import agents


CHECKPOINTS = 'checkpoints'


class Player(object):

    def __init__(self, env, monitor='output/', seed=None):
        self.env = env
        self.agents = {'universe': agents.A3C(env, monitor+'universe/',
                                              CHECKPOINTS+'/universe/'+env+'/', 1),
                       'tensorpack': agents.TPAgent(env, monitor+'tensorpack/',
                                              CHECKPOINTS+'/tensorpack/'+env+'/'+env, 1),
                       'random': agents.RandomAgent()}
        self.seed = seed
        self.best = ''
        
        
    def choose(self, num_episodes_eval=100):
        scores = {}
        for agent in self.agents.keys():
            scores[agent] = self.agents[agent].play(num_episodes_eval,
                                                    env=self.env,
                                                    record=False,
                                                    seed=self.seed)
            
        self.best = max(scores, key=scores.get)


    def choose_and_record(self, num_episodes_eval=100, num_episodes_run=100):
        scores = {}
        for agent in self.agents.keys():
            scores[agent] = self.agents[agent].play(num_episodes_eval,
                                                    env=self.env,
                                                    record=False,
                                                    seed=self.seed)
            
        self.best = max(scores, key=scores.get)
        self.agents[max(scores, key=scores.get)].play(num_episodes_run,
                                                      env=self.env,
                                                      record=True,
                                                      seed=self.seed)

    def upload(self, outputm, api_key=''):
        self.agents[max(scores, key=scores.get)].do_submit(output, api_key)
