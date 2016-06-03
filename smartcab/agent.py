import random
import os
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# Table for the Q charts. Creating a class to store the table.
class QChart(object):
    """ Provides a table chart for the q values to be documented from the
       experience learned from the agent.
    """
    def __init__(self):
        """ Loads the Q_value to be registered as a dictionary.
        """
        self.Q_value = dict()
        
    def get(self, state, action):
        """ Receives Q_value based on state and action of the agent
        
            Args:
              state: Position of the agent.
              action: direction of the agent.
        """
        key = (state, action)
        return self.Q_value.get(key)
    
    def set(self, state, action, q_value):
        """ Changing up the q_value for a new value.
        
            Args:
              state: Position of the agent.
              action: direction of the agent.
              q_value: computed q_value to replace the current value.
        """
        key = (state,action)
        self.Q_value[key] = q_value
    
    def report(self):
        """ Reporting the result of the Q chart.
        """
        for k, v in self.Q_value.items():
            print k,v

class Qlearn(Agent):
    """ Reference from the agent class, a q learning agent that is able to
        learn and update based on its previous behavior.
    Attribute:
      Q: value for the q table.
      episilon = probability of the random move
      alpha = learning rate
      gamma = memory and discount of the max Q.
    """
    
    def __init__(self, epsilon = 1, alpha = 1, gamma =1):
        self.Q = QChart()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.possible_actions = Environment.valid_actions
        
    def Q_move(self, state):
        """ With the given state, provide the best actions to go.
        
        Args:
          state = current state the agent is in
        """
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            q = [self.Q.get(state,action) for action in self.possible_actions]
            max_q = max(q)
            
            if q.count(max_q) > 1:
                best_actions = [i for i in range(len(self.possible_actions)) if q[i] == max_q]
                action_idx = random.choice(best_actions)
            else:
                action_idx = q.index(max_q)
            action = self.possible_actions[action_idx]
        return action
    
    def Q_learn(self, state, action, reward, new_q):
        q = self.Q.get(state, action)
        if q is None:
            q = reward
        else:
            q+= self.alpha * new_q
        self.Q.set(state,action,q)
    
    def Q_post(self, state, action, next_state, reward):
        """ With the given result, update and reload the q value
        
        Args:
          state: current state it is in.
          actions: action that it took.
          next_state: next state it goes into.
          reward: rewards it has received.
        """
        
        q_value = [self.Q.get(next_state,a) for a in self.possible_actions]
        future_rewards = max(q_value)
        if future_rewards is None:
            future_rewards  = 0.0
        self.Q_learn(state,action,reward, reward - self.gamma * future_rewards)
        self.Q.report()
    
class QLearningAgent(Agent):
    """ An Agent that is learned to drive based on the Q-Learning
        techniques
        
    color: color of the car
    planner: plan on how it gets to the next point
    ai: QLearn suggested route
    """
    
    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)
        self.color = 'black'
        self.planner = RoutePlanner(self.env, self)
        self.possible_actions = Environment.valid_actions
        self.ai = Qlearn(epsilon=0.05, alpha=0.1, gamma=0.9)
    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        
    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()
        
        inputs = self.env.sense(self)
        inputs = inputs.items()
        deadline = self.env.get_deadline(self)
        
        self.state = (inputs[0], inputs[1], inputs[3], self.next_waypoint)
        
        action = self.ai.Q_move(self.state)
        
        reward = self.env.act(self, action)
        
        inputs2 = self.env.sense(self)
        inputs2 = inputs2.items()
        
        next_state = (inputs2[0], inputs2[1], inputs[3], self.next_waypoint)
        
        self.ai.Q_post(self.state, action, next_state, reward)
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)
    
def run():
    """ Run teh agent in a finite of trials"""
    
    e = Environment()
    a = e.create_agent(QLearningAgent)
    e.set_primary_agent(a,enforce_deadline=True)
    
    sim = Simulator(e, update_delay=0.001)
    
    sim.run(n_trials=1)

if __name__ == '__main__':
    run()
    