"""
Quantization method for quantizing the point cloud
"""


class Quantization:
    def __init__(self):
        self.methods = []

    def add_method(self, method):
        self.methods.append(method)

    def quantize(self, tree, pro):
        self.quantize_pending_leaves(tree,pro)
        self.quantize_best_leaf(tree,pro)

    def quantize_pending_leaves(self, tree, pro):
        """Test each quantization method on new leaves"""
        store_pending_leaves = tree.leaves_pending
        #print("---> pending leaves are: ", store_pending_leaves)
        for leaf_id in store_pending_leaves:
            # get the leaf node corresponding to the leaf_id in pending leaves
            leaf = tree.get_node(leaf_id)
            # Save input point data to restore between each quantization
            source_data = tree.trainingSequence.retrieve(leaf.inputs)
            # Init data to store the best quantization method
            best_quantized_node = None
            # Test each method one by one
            for id in range(len(self.methods)):
                # print("the leaf being quantized is: ", leaf_id, " using the method: ", id)
                # Clone the leaf
                leaf_clone = leaf.quantization_clone()
                leaf_clone.methodID = id
                leaf_clone.b = self.methods[id].b
                leaf_clone.isLeaf = False

                # Apply quantization on leaf clone
                self.methods[id].quantize_node(tree.trainingSequence, leaf_clone)

                # generate the partition state corresponding to quantization of this leaf
                # we access the partition state of parent and assign 1 to region with representant of current leaf
                par_state = []
                # check root
                if leaf_id == 0:
                    par_state = [0]

                else:
                    parent_state = leaf.parent.partition_state

                    # partition state will have 1 at index = representant of current leaf and 0 at other index

                    rep = leaf.representant
                    for i in range(len(parent_state)):
                        if i == rep:
                            par_state.append(1)
                        else:
                            par_state.append(0)

                # Compute the new lambda and assign it to lambda score of leaf clone
                leaf_clone.update_lambda(tree.trainingSequence,par_state,pro)

                # print("leaf clone with id: ",leaf_id," , for method no: ",leaf_clone.methodID," , has the lambda score = ",leaf_clone.lambdaScore)

                # Check if quantization is better than the already tested ones
                if best_quantized_node == None or leaf_clone.lambdaScore > best_quantized_node.lambdaScore:
                    # Check if delta distortion is decreasing from parent to children
                    #if leaf_clone.deltaDistortion > 0 and leaf_clone.lambdaScore != 0:
                    best_quantized_node = leaf_clone

                # Restore data for next quantization
                tree.trainingSequence.set(leaf.inputs, source_data)
                # ?????????? leaf.inputs and source data are one at the same.....

            # If a quantization was found
            if best_quantized_node != None:

                # Store node results to this node
                leaf.save_quantization(best_quantized_node)
                # this leaf has lambda score calculated while assuming its quantization in previous loop

                # Insert the lambda
                tree.insert_lambda(leaf.lambdaScore, leaf.id)

        # restore the pending leaves to [] since all of them have been traversed now
        tree.leaves_pending = []

    def quantize_best_leaf(self, tree, pro):
        # update lambda score of all the contenders before determining best leaf

        """Quantize best lambda node"""
        #print("lambda_sorted : ", tree.lambda_sorted)

        id = tree.pop_max_lambda()

        if id == -1:
            return

        leaf = tree.get_node(id)

        # Add new leaves
        for child in leaf.children:
            # Apply quantization to inputs
            self.methods[leaf.methodID].apply_quantization(tree.trainingSequence, child)
            # add the newly created children to the leaf
            tree.add_leaf(child)

        # generate a partition state for the leaf based on which quantization method was used to split it
        leaf.partition_state = [0] * ((leaf.methodID + 2) ** 3)


        # now the leaf has become a node
        leaf.isLeaf = False
        # print ("initial dist: ",tree.distortion)
        # print("initial rate: ",tree.rate)

        # Update the distortion and rate of the tree
        tree.distortion -= leaf.deltaDistortion
        #print("dist: ",tree.distortion)
        tree.rate += leaf.deltaRate

        #print("rate: ",tree.rate)

        # get the parent of the best leaf to make changes in the partition state and update probability array
        best_parent = leaf.parent
        # when the best leaf is root during first quantization
        if best_parent == None:
            # the number of zeroes for partition state of root will depend on quantization method used
            leaf.partition_state = [0] * ((leaf.methodID+2) ** 3)
            return

        # partition state of parent before splitting the best leaf
        old_state = best_parent.partition_state

        # for nodes except root we use representant of best leaf to calculate partition state of best parent
        rep = leaf.representant

        # after quantization
        new_state = []
        # we will append a new "1" only at the index corresponding to representant of best leaf
        for i in range(len(old_state)):
            if i==rep:
                new_state.append(1)
            else:
                new_state.append(old_state[i])

        # make changes to prob array since new state is being added and old one if not all zeroes will be subtracted
        pro.sub_add(old_state,new_state)

        #print("old state is:- ",old_state," new state is:- ",new_state)
        # print("the probability table is:- ", pro.probability_table)


        # if the total partitions do not change then probability_occurencey of all remaining partition states is the same
        if pro.get_index(old_state) != 0:
            best_parent.partition_state = new_state
            return

        # we update lambda score of all leaves in lambda_sorted only when there is change in total number of partition state
        else:
            best_parent.partition_state = new_state
            new_lambda_sorted = {}
            # obtain lambda scores from lambda sorted
            for i in tree.lambda_sorted:
                # get leaves having the same lambda score
                same_lambda_leaves = tree.lambda_sorted[i]
                # now update lambda score of each leaf seperately
                for j in same_lambda_leaves:
                    n = tree.get_node(j)
                    n.update_lambda_score (pro)
                    new_lambda_score = n.lambdaScore
                    if new_lambda_score in new_lambda_sorted:
                        new_lambda_sorted[new_lambda_score].append(n.id)
                    else:
                        new_lambda_sorted[new_lambda_score] = [n.id]

            # we create new lambda sorted only when an increase in total number of partition states occur
            #print("new lambda sorted = ",new_lambda_sorted)
            tree.lambda_sorted = new_lambda_sorted
            return





                




    

