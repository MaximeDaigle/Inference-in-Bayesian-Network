import random


class Node:

    def __init__(self, name, parents, probability_table):
        self.name = name
        self.parents = parents
        self.prob_table = probability_table
        self.value = None

    def conditional_probability(self):
        known = [p.value for p in self.parents if p.value is not None]
        if len(known) != len(self.parents):
            print('manque des valeurs pour calculer probabilites conditionnelles')
            return None
        if len(known) == 1:
            return self.prob_table[known[0]]
        return self.prob_table[tuple(known)]


burglary = Node('Burglary', [],
                {(): 0.001})

earthquake = Node('Earthquake', [],
                  {(): 0.002})

alarm = Node('Alarm', [burglary, earthquake],
             {(True, True): 0.95,
              (True, False): 0.94,
              (False, True): 0.29,
              (False, False): 0.001})

john = Node('JohnCalls', [alarm],
            {(True): 0.9,
             (False): 0.05})

mary = Node('MaryCalls', [alarm],
            {(True): 0.7,
             (False): 0.01})

bayes_network = [burglary, earthquake, alarm, john, mary]

def get_node(bayes_network, name_node):
    for node in bayes_network:
        if node.name == name_node:
            return node
    print('Node', name_node, 'not in bayes network')
    return None

def prior_sample(bayes_network):
    """
    A sampling algorithm that generates events from a Bayesian network.
    Each variable is sampled according to the conditional distribution given the values
    already sampled for the variable’s parents
    """
    for node in bayes_network:
        node.value = random.random() < node.conditional_probability()

def consistent(bayes_network, evidence):
    #verifie si bayes network fonctionne avec nos evidences
    for e in evidence:
        if get_node(bayes_network, e[0]).value != e[1]:
            return False
    return True

def rejection_sampling(X,evidence, N):
    """
    :param X: la probabiliter rechercher est celle de la variable X
    :param evidence: valeur observer pour des variables
    :param N: nombre d'echantillonage a faire
    :return: estimation de la probabilite de X sachant e
    """
    counts = [0,0] # compte le nb de fois ou X = False, X = True
    for i in range(N):
        prior_sample(bayes_network)
        if consistent(bayes_network, evidence):
            counts[int(get_node(bayes_network,X).value)] += 1
    if counts[0] == 0 or counts[1] == 0:
        # restart sampling because missed rare event
        return rejection_sampling(X,evidence,N)
    return counts[1] / (counts[0] + counts[1])


def likelihood_weighting(X, evidence, N):
    """
    :param X: query variable
    :param evidence: observed values
    :param N: total number of samples generated
    """
    #W: a vector of weighted counts for each value of X , initially zero
    W = [0,0]
    for j in range(N):
        w = weighted_sample(bayes_network,evidence)
        x = get_node(bayes_network, X).value
        W[int(x)] += w
    if W[1] == 0 or W[0] == 0:
        #restart sampling because missed rare event
        return likelihood_weighting(X,evidence,N)
    return W[1] / (W[1] + W[0])

def weighted_sample(bayes_network, evidence):
    """
    each nonevidence variable is sampled according to the conditional distribution given
    the values already sampled for the variable’s parents, while a weight is
    accumulated based on the likelihood for each evidence variable.
    """
    w = 1
    #initialize an event from the evidence
    evidence_nodes = set()
    #fixes evidence variables in the event
    for e in evidence:
        e_node = get_node(bayes_network, e[0])
        e_node.value = e[1]
        evidence_nodes.add(e_node)

    #initiliaze the other nodes, starting from the parents
    o_nodes = set(bayes_network) - evidence_nodes
    other_nodes = [node for node in bayes_network if node in o_nodes]
    for node in other_nodes:
        node.value = random.random() < node.conditional_probability()
    #accumulate weight for each evidence variable
    for e_node in evidence_nodes:
        w = w * e_node.conditional_probability()

    return w




if __name__ == "__main__":
    for N in [100,1000,10000]:
        p = rejection_sampling('Burglary',[('JohnCalls', True), ('Earthquake', False)], N)
        print('Probability with rejecting sampling', p ,'(avec', N, 'essaies)')

    print()

    for N in [100,1000,10000]:
        p = likelihood_weighting('Burglary',[('JohnCalls', True), ('Earthquake', False)], N)
        print('Probability with likelihood weighting', p ,'(avec', N, 'essaies)')