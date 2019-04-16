import os
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.utils.data as data

import Constants
from Tree import Tree

# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        print("\t* init new SICKDataset:", path, vocab, num_classes)
        self.vocab = vocab
        self.num_classes = num_classes
        self.lsentences = self.read_sentences(os.path.join(path, 'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path, 'b.toks'))
        
        self.ltrees = self.read_trees(os.path.join(path, 'a.parents'))
        self.rtrees = self.read_trees(os.path.join(path, 'b.parents'))
        
        self.lparents = self.read_parents(os.path.join(path, 'a.parents'))
        self.rparents = self.read_parents(os.path.join(path, 'b.parents'))

        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))
        self.size = self.labels.size(0)
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        lparents = deepcopy(self.lparents[index])
        rparents = deepcopy(self.rparents[index])
        label = deepcopy(self.labels[index])
        return (ltree, lsent, lparents, rtree, rsent, rparents, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        print("line:", line, "parents:", parents, "# parents:", len(parents))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            #print("i:", i, "; trees' keys:", trees.keys(), "; parents[i-1]:", parents[i-1])

            # if we haven't processed it, and its listed parent is actually sensical (has a parent)
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                #print("\tin if; idx = i = ", i)

                # follows parent links, starting with index i
                while True:
                    #print("\t\tin while loops")
                    parent = parents[idx - 1]
                    #print("\t\t\tparent:", parent)
                    if parent == -1:
                        #print("*** parent is -1!")
                        break
                    tree = Tree()
                    #print("\t\t\tprev:", prev)
                    if prev is not None:
                        #print("\t\t\tprev isn't None, so adding it as a children to Tree")
                        tree.add_child(prev)
                    
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    #print("\t\t\ttrees:", trees)
                    #print("\t\t\ttree = ", tree, "idx:", tree.idx)
                    if parent - 1 in trees.keys(): # simply checks if we should point its parent to the current node
                        #print("\t\t\t\tparent -1 is in trees!")
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0: # make it the root
                        #print("\t\t\t\tparent = 0")
                        root = tree
                        break
                    else:
                        #print("\t\t\t\tset prev to be = tree; idx = parent", idx, "=", parent)
                        prev = tree
                        idx = parent
        print("# trees:", len(trees.keys()))

        all_nodes = []
        self.dfs(root, all_nodes)
        print("all_nodes:", len(all_nodes))
        exit()
        return root

    def dfs(self, cur_node, all_nodes):
        if cur_node in all_nodes:
            print("oh shit")
            exit()
        all_nodes.append(cur_node)

        for idx in range(cur_node.num_children):
            self.dfs(cur_node.children[idx], all_nodes)


    def read_parents(self, filename):
        with open(filename, 'r') as f:
            parents = [list(map(int, line.split())) for line in f.readlines()]
        return parents

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float, device='cpu')
        return labels
