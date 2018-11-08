import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from scipy.special import logsumexp 
 
 
class BinaryGraph: 
    def __init__(self, nodes, edges, node_potentials, edge_potentials): 
        """ 
        nodes: a set of labels for the nodes of the graph 
        edges: a set of pairs of nodes 
        node_potentials: a dictionary mapping each node i to [psi_i(0), psi_i(1)] 
        edge_potentials: a dictionary mapping each edge (i, j) to [[psi_ij(0,0), psi_ij(0,1)], [psi_ij(1,0), psi_ij(1,1)]] 
        """ 
        self.nodes = nodes 
        self.edges = edges 
        self.node_potentials = node_potentials.copy() 
        self.edge_potentials = edge_potentials.copy() 
        self.edge_potentials.update({(j, i): potential for (i, j), potential in edge_potentials.items()}) 
 
        self.log_node_potentials = {node: np.log(potential) for node, potential in node_potentials.items()} 
        self.log_edge_potentials = {edge: np.log(potential) for edge, potential in self.edge_potentials.items()} 
 
    def initialize_messages(self): 
        messages_meta = pd.DataFrame(index=range(len(self.edges)*2), columns=['source', 'target']) 
        log_messages = np.zeros([len(self.edges)*2, 2]) 
        for i, (node1, node2) in enumerate(self.edges): 
            messages_meta.loc[2*i].source = node1 
            messages_meta.loc[2*i].target = node2 
            log_messages[2*i] = np.array([0, 0]) 
 
            messages_meta.loc[2*i+1].source = node2 
            messages_meta.loc[2*i+1].target = node1 
            log_messages[2*i+1] = np.array([0, 0]) 
        return messages_meta, log_messages 
 
    def run_bp(self): 
        messages_meta, log_messages = self.initialize_messages() 
        count = 0 
        while count < 30: 
            print("Here")
            new_log_messages = log_messages.copy() 
            for ix, log_message in tqdm(enumerate(log_messages), total=log_messages.shape[0]): 
                message_meta = messages_meta.loc[ix] 
 
                log_source_potential = self.log_node_potentials[message_meta.source] 
                log_edge_potential = self.log_edge_potentials[(message_meta.source, message_meta.target)] 
                messages_to_source_ixs = (messages_meta.target == message_meta.source) & (messages_meta.source != message_meta.target) 
                log_messages_to_source = log_messages[messages_to_source_ixs] 
                total_log_source_potential = log_messages_to_source.sum(axis=0) + log_source_potential 
 
                new_log_message_unnormalized = logsumexp(total_log_source_potential[:, np.newaxis] + log_edge_potential, axis=0) 
                new_log_messages[ix] = new_log_message_unnormalized - logsumexp(new_log_message_unnormalized) 
            print(new_log_messages.mean()) 
            avg_change = np.mean(np.abs(log_messages - new_log_messages)) 
            #converged = avg_change < .01 
            print(avg_change) 
            log_messages = new_log_messages 
            count += 1
 
        marginals = {} 
        for node in self.nodes: 
            messages_to_node_ixs = messages_meta.target == node 
            messages_to_node = log_messages[messages_to_node_ixs] 
            unnormalized_log_marginals = self.log_node_potentials[node] + messages_to_node.sum(axis=0) 
            normalized_log_marginals = unnormalized_log_marginals - logsumexp(unnormalized_log_marginals) 
            marginals[node] = np.exp(normalized_log_marginals) 
 
        return marginals 
 
 
def marginals_to_image(marginals, length, height): 
    img = np.zeros([length, height]) 
    for node, marginal in marginals.items(): 
        i, j = node 
        img[i, j] = marginal[1] 
    return img 