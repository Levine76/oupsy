# this class contains the table of probabilities for all partition states
class Probability:
    def __init__(self):
# each index of probability_table has (partition state, counter, probability occurency)
        self.probability_table = [[]] * 37
# total_states is the sum of occurency of each partition state
        self.total_states = 0
# j is the counter for number of voronoi cell leaves
        self.j = 0

# this function adds index of partition state in table
    def indexing_states (self):
        for i in range(len(self.probability_table)):
            if self.probability_table[i] == []:
                continue
            else:
                for j in range(len(self.probability_table[i])):
                    self.probability_table[i][j].append([i,j])



    # checker gives true when the partition_state has all zeros, ie: no splitting has occured
    def checker(self, par):
        for i in par:
            if i == 1:
                return False
        return True

    # hash function
    def get_index(self, par):
        counter = 0
        for i in range(len(par)):
            if par[i] == 1:
                counter += i
        return counter % 37


    # returns the probability occurency of a particular partition state
    def get_probability (self,par):
        if self.checker(par):
            return 0
        index = self.get_index (par)
        for state in self.probability_table[index]:
            if state[0] == par:
                return state[2]


    # we have to update all the values of probabilities when total number of partition states changes
    def update_probability (self):
        if self.total_states == 0:
            self.probability_table = [[]] * 37
            return
        for chain in self.probability_table:
            if chain == []:
                continue
            for state in chain:
                state[2] = state[1] / self.total_states
                if state[2] == 0:
                    # remove the state whose counter is zero
                    chain.remove(state)
                    # keep updating other states in the chain even after removing
                    for state in chain:
                        state[2] = state[1] / self.total_states

    # increse the counter of existing state or create a new state
    def add_state (self,par):
        if self.checker(par):
            return
        index = self.get_index(par)
        if self.probability_table[index] == []:
            self.total_states += 1
            new_chain = [[par,1,0]]
            self.probability_table[index] = new_chain
            self.update_probability()
            return
        for state in self.probability_table[index]:
            if state[0] == par:
                state[1] += 1
                self.total_states += 1
                self.update_probability()
                return
        self.total_states += 1
        new_state = [par,1, 0]
        self.probability_table[index].append(new_state)
        self.update_probability()

    # subtract the counter of existing state
    def sub_state (self,par):
        if self.checker(par):
            return
        index = self.get_index(par)
        for state in self.probability_table[index]:
            if state[0] == par:
                state[1] -= 1
                self.total_states -= 1
                self.update_probability()



    # it adds one new state (par1) to the prob array and removes one old state (par2)
    def sub_add (self,par1,par2):
        index1 = self.get_index(par1)
        index2 = self.get_index(par2)
        # check if the state being subtracted is all zeroes
        if self.checker(par1):
            self.add_state(par2)
            return
        # perform sub & add operations
        self.sub_state(par1)
        self.add_state(par2)





