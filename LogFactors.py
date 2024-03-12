# pylint: skip-file
import numpy as np
from DiscreteFactors import Factor

class LogFactor(Factor):
    def join(self, other):
        '''
        Usage: `new = f.join(g)` where f and g are Factors
        This function multiplies two factors.
        '''
        # confirm that any shared variables have the same outcomeSpace
        for var in set(other.domain).intersection(set(self.domain)):
            if self.outcomeSpace[var] != other.outcomeSpace[var]:
                raise IndexError('Incompatible outcomeSpaces. Make sure you set the same evidence on all factors')
        # extend current domain with any new variables required
        new_dom = list(self.domain) + list(set(other.domain) - set(self.domain)) 
        # to prepare for multiplying arrays, we need to make sure both arrays have the correct number of axes
        self_t = self.table
        other_t = other.table
        for _ in set(other.domain) - set(self.domain):
            self_t = self_t[..., np.newaxis]     
        for _ in set(self.domain) - set(other.domain):
            other_t = other_t[..., np.newaxis]
        # And we need the new axes to be transposed to the correct location
        old_order = list(other.domain) + list(set(self.domain) - set(other.domain)) 
        new_order = []
        for v in new_dom:
            new_order.append(old_order.index(v))
        other_t = np.transpose(other_t, new_order)
        # Now that the arrays are all set up, we can rely on numpy broadcasting to work out which numbers need to be added.
        new_table = self_t + other_t
        # The final step is to create the new outcomeSpace
        new_outcomeSpace = self.outcomeSpace.copy()
        new_outcomeSpace.update(other.outcomeSpace)
        return self.__class__(tuple(new_dom), new_outcomeSpace, table=new_table)

    def marginalize(self, var):
        raise NotImplementedError("We don't usually marginalize log factors")
    def normalize(self):
        raise NotImplementedError("We don't usually normalize log factors")