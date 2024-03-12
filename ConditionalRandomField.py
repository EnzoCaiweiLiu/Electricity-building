# pylint: skip-file
import numpy as np
from LogFactors import LogFactor

class ConditionalRandomField():
    def __init__(self, first_state, tFactor, variable_remap, uniform_dist=False):
        if uniform_dist:
            self.state = LogFactor(first_state.domain, first_state.outcomeSpace)
        else:
            self.state = first_state
        self.transition = tFactor

        self.remap = variable_remap

        self.history = []
        self.prev_history = []

    def viterbiStep(self, observationFactor):

        # confirm that state and emission each have 1 variable 
        assert len(self.state.domain) == 1
        assert len(self.transition.domain) == 2

        # get state and evidence var names (to be marginalized and maximised out later)
        state_var_name = self.state.domain[0]

        # join with transition factor
        f = self.state*self.transition

        # maximize out old state vars, leaving only new state vars
        f, prev = f.maximize(state_var_name, return_prev=True)
        self.prev_history.append(prev)

        # remap variable to it's original name
        f.domain = tuple(self.remap[var] for var in f.domain)

        # join observation factor with state factor
        f = f*observationFactor

        self.state = f 

        self.history.append(self.state)

        return self.state

    def viterbiBatch(self, observationFactorList):
        '''
        emissionEviList: A list of dictionaries, each dictionary containing the evidence for that timestep. 
                         Use `None` if no evidence for that timestep
        '''
        for observationFactor in observationFactorList:
            # select evidence for this timestep
            self.viterbiStep(observationFactor)
        return self.history

    def traceBack(self):
        # get most likely outcome of final state
        index = np.argmax(self.history[-1].table)
        
        # Go through "prev_history" in reverse
        indexList = []
        for prev in reversed(self.prev_history):
            indexList.append(index)
            index = prev[index]
        indexList = reversed(indexList)

        # translate the indicies into the outcomes they represent
        mleList = []
        stateVar = self.state.domain[0]
        for idx in indexList:
            mleList.append(self.state.outcomeSpace[stateVar][idx]) 
        return mleList