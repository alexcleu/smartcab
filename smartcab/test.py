from agent import *

def run():
    """ Run teh agent in a finite of trials"""

    e = Environment()
    a = e.create_agent(QLearningAgent)
    e.set_primary_agent(a,enforce_deadline=False)

    sim = Simulator(e, update_delay=0.00001)
    #TODO Write a test for the best tuning...

    sim.run(n_trials=1)

if __name__ == '__main__':
    run()
