class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """
    def __init__(self, agents):
        self.k = len(agents)
        self.agents = agents

    def pull(self, agent):
        return agent.play()
        

class Player(object):
    """
    A Player is able to play one of a set of agents at each trial. The
    agent is chosen using a strategy based on the history of prior plays
    and outcome observations.
    """
    def __init__(self, bandit, policy, prior=0, gamma=None):
        self.policy = policy
        self.k = bandit.k
        self.prior = prior
        self.gamma = gamma
        self._reward_estimates = prior*np.ones(self.k)
        self.agent_attempts = np.zeros(self.k)
        self.t = 0
        self.last_agent = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        """
        Resets the player's memory to an initial state.
        """
        self._reward_estimates[:] = self.prior
        self.agent_attempts[:] = 0
        self.last_agent = None
        self.t = 0

    def choose(self):
        agent = self.policy.choose(self)
        self.last_agent = agent
        return agent

    def observe(self, reward):
        self.agent_attempts[self.last_agent] += 1

        if self.gamma is None:
            g = 1 / self.agent_attempts[self.last_agent]
        else:
            g = self.gamma
        q = self._reward_estimates[self.last_agent]

        self._reward_estimates[self.last_agent] += g*(reward - q)
        self.t += 1

    @property
    def reward_estimates(self):
        return self._reward_estimates


class Environment(object):
    def __init__(self, bandit, player, num_episodes, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.player = player
        self.label = label

    def reset(self):
        self.player.reset()

    def run(self, trials=100, experiments=10):
        scores = np.zeros((trials, 1))

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                agent = player.choose()
                reward = self.bandit.pull(agent, num_episodes)
                player.observe(reward)

        return scores / experiments, self.player._reward_estimates


    """
    def plot_results(self, scores):
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average Reward')
        plt.legend(self.player, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(optimal * 100)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.player, loc=4)
        sns.despine()
        plt.show()
    """

        

class Policy(object):
    """
    A policy prescribes an agent to be chosen based on the memory of an player.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, player):
        return 0

class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random agent with probability
    epsilon and run the best apparent agent with probability 1-epsilon. If
    multiple agents are tied for best choice, then a random agent from that
    subset is selected.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, player):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(player.reward_estimates))
        else:
            agent = np.argmax(player.reward_estimates)
            check = np.where(player.reward_estimates == agent)[0]
            if len(check) == 0:
                return agent
            else:
                return np.random.choice(check)

class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each agent which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, agent):
        exploration = np.log(player.t+1) / player.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = player.value_estimates + exploration
        agent = np.argmax(q)
        check = np.where(q == agent)[0]
        if len(check) == 0:
            return agent
        else:
            return np.random.choice(check)



