import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from graphnnSiamese import graphnn
import cPickle as pickle

def get_f_name(DATA, SF, CM, OP, VS):
	F_NAME = []
	for sf in SF:
		for cm in CM:
			for op in OP:
				for vs in VS:
					F_NAME.append(DATA+sf+cm+op+vs+".txt")
	return F_NAME


def get_f_dict(F_NAME):
	name_num = 0
	name_dict = {}
	for f_name in F_NAME:
		cur_f = open(f_name, "r")
		for line in cur_f:
			info = line.strip().split(' ')
			if (len(info) == 2):
				if (len(info[1].split('||')) > 1):
					info[1] = info[1].split('||')[0]
				if (info[1] not in name_dict):
					name_dict[info[1]] = name_num
					name_num += 1
	return name_num, name_dict

class graph(object):
	def __init__(self, node_num = 0, label = None, name = None):
		self.node_num = node_num
		self.label = label
		self.name = name
		self.features = []
		self.succs = []
		self.preds = []
		if (node_num > 0):
			for i in range(node_num):
				self.features.append([])
				self.succs.append([])
				self.preds.append([])
				
	def add_node(self, feature = []):
		self.node_num += 1
		self.features.append(feature)
		self.succs.append([])
		self.preds.append([])
		
	def add_edge(self, u, v):
		self.succs[u].append(v)
		self.preds[v].append(u)

	def toString(self):
		ret = '{} {}\n'.format(self.node_num, self.label)
		for u in range(self.node_num):
			for fea in self.features[u]:
				ret += '{} '.format(fea)
			ret += str(len(self.succs[u]))
			for succ in self.succs[u]:
				ret += ' {}'.format(succ)
			ret += '\n'
		return ret

		
def read_graph(F_NAME, FUNC_NAME_DICT, FEATURE_DIM):
	graphs = []
	classes = []
	if FUNC_NAME_DICT != None:
		for f in range(len(FUNC_NAME_DICT)):
			classes.append([])

	for f_name in F_NAME:
		cur_f = open(f_name, "r")
		g_num = int(cur_f.readline())
		for g_id in range(g_num):
			## Read in a single graph.
			info = cur_f.readline().strip().split(' ')
			n_num = int(info[0])
			if FUNC_NAME_DICT != None:
				fname = info[1]
				if (len(fname.split('||')) > 1):
					fname = fname.split('||')[0]
				label = FUNC_NAME_DICT[fname]
				classes[label].append( len(graphs) )    ## Record which graphs that each class contains.
			else:
				label = info[1]
			cur_graph = graph(n_num, label, info[1])
			for u in range(n_num):
				info = cur_f.readline().strip().split(' ')
				cur_graph.features[u] = np.array(map(float, info[:FEATURE_DIM]))
				succ_num = int(info[FEATURE_DIM])
				for strv in info[FEATURE_DIM+1:]:
					cur_graph.add_edge(u, int(strv))
				
			graphs.append(cur_graph)
			

	if FUNC_NAME_DICT != None:
		return graphs, classes
	else:
		return graphs

def read_added_pair(fname, fea_dim):
	#Read graphs
	Gs = read_graph([fname+'.graph'], None, fea_dim)

	#Generate pairs
	id_file = open(fname+'.id', 'r')
	added_pair_data = []   #[(X1, X2, m1, m2, y)]
	for line in id_file:
		i1, i2, label = map(int, line.strip().split(' '))

		g1 = Gs[i1]
		X1 = np.zeros((1, g1.node_num, fea_dim))
		m1 = np.zeros((1, g1.node_num, g1.node_num))
		for u in range(g1.node_num):
			X1[0, u, :] = np.array( g1.features[u] )
			for v in g1.succs[u]:
				m1[0, u, v] = 1

		g2 = Gs[i2]
		X2 = np.zeros((1, g2.node_num, fea_dim))
		m2 = np.zeros((1, g2.node_num, g2.node_num))
		for u in range(g2.node_num):
			X2[0, u, :] = np.array( g2.features[u] )
			for v in g2.succs[u]:
				m2[0, u, v] = 1

		added_pair_data.append([X1, X2, m1, m2, [label]])

	return added_pair_data


def partition_data(Gs, classes, partitions, perm):
	C = len(classes)
	st = 0.0
	ret = []
	for part in partitions:
		cur_g = []
		cur_c = []
		ed = st + part * C
		for cls in range(int(st), int(ed)):
			prev_class = classes[perm[cls]]
			cur_c.append([])
			for i in range(len(prev_class)):
				cur_g.append(Gs[prev_class[i]])
				cur_g[-1].label = len(cur_c)-1
				cur_c[-1].append(len(cur_g)-1)

		ret.append(cur_g)
		ret.append(cur_c)
		st = ed

	return ret


def generate_epoch_pair(Gs, classes, M, output_id = False, load_id = None):
	epoch_data = []
	id_data = []   # [ ( [(G0,G1),(G0,G1)] , [(G0,H0),(G0,H0)] ) , ... ]

	if load_id is None:
		st = 0
		while st < len(Gs):
			if output_id:
				X1, X2, m1, m2, y, pos_id, neg_id = get_pair(Gs, classes, M, st=st, output_id=True)
				id_data.append( (pos_id, neg_id) )
			else:
				X1, X2, m1, m2, y = get_pair(Gs, classes, M, st=st)
			epoch_data.append( (X1,X2,m1,m2,y) )
			st += M
	else:   ## Load from previous id data
		id_data = load_id
		for id_pair in id_data:
			X1, X2, m1, m2, y = get_pair(Gs, classes, M, load_id=id_pair)
			epoch_data.append( (X1, X2, m1, m2, y) )


	if output_id:
		return epoch_data, id_data
	else:
		return epoch_data


def get_pair(Gs, classes, M, st = -1, output_id = False, load_id = None):
	if load_id is None:
		C = len(classes)

		if (st + M > len(Gs)):
			M = len(Gs) - st
		ed = st + M

		pos_ids = [] # [(G_0, G_1)]
		neg_ids = [] # [(G_0, H_0)]

		for g_id in range(st, ed):
			g0 = Gs[g_id]
			cls = g0.label
			tot_g = len(classes[cls])
			if (len(classes[cls]) >= 2):
				g1_id = classes[cls][np.random.randint(tot_g)]
				while g_id == g1_id:
					g1_id = classes[cls][np.random.randint(tot_g)]
				pos_ids.append( (g_id, g1_id) )


			cls2 = np.random.randint(C)
			while (len(classes[cls2]) == 0) or (cls2 == cls):
				cls2 = np.random.randint(C)

			
			tot_g2 = len(classes[cls2])
			h_id = classes[cls2][np.random.randint(tot_g2)]
			neg_ids.append( (g_id, h_id) )
	else:
		pos_ids = load_id[0]
		neg_ids = load_id[1]
		
	M_pos = len(pos_ids)
	M_neg = len(neg_ids)
	M = M_pos + M_neg

	maxN1 = 0
	maxN2 = 0
	for pair in pos_ids:
		maxN1 = max(maxN1, Gs[pair[0]].node_num)
		maxN2 = max(maxN2, Gs[pair[1]].node_num)
	for pair in neg_ids:
		maxN1 = max(maxN1, Gs[pair[0]].node_num)
		maxN2 = max(maxN2, Gs[pair[1]].node_num)


	feature_dim = len(Gs[0].features[0])
	X1_input = np.zeros((M, maxN1, feature_dim))
	X2_input = np.zeros((M, maxN2, feature_dim))
	node1_mask = np.zeros((M, maxN1, maxN1))
	node2_mask = np.zeros((M, maxN2, maxN2))
	y_input = np.zeros((M))
	
	for i in range(M_pos):
		y_input[i] = 1
		g1 = Gs[pos_ids[i][0]]
		g2 = Gs[pos_ids[i][1]]
		for u in range(g1.node_num):
			X1_input[i, u, :] = np.array( g1.features[u] )
			for v in g1.succs[u]:
				node1_mask[i, u, v] = 1
		for u in range(g2.node_num):
			X2_input[i, u, :] = np.array( g2.features[u] )
			for v in g2.succs[u]:
				node2_mask[i, u, v] = 1


		
	for i in range(M_pos, M_pos + M_neg):
		y_input[i] = -1
		g1 = Gs[neg_ids[i-M_pos][0]]
		g2 = Gs[neg_ids[i-M_pos][1]]
		for u in range(g1.node_num):
			X1_input[i, u, :] = np.array( g1.features[u] )
			for v in g1.succs[u]:
				node1_mask[i, u, v] = 1
		for u in range(g2.node_num):
			X2_input[i, u, :] = np.array( g2.features[u] )
			for v in g2.succs[u]:
				node2_mask[i, u, v] = 1
	if output_id:
		return X1_input, X2_input, node1_mask, node2_mask, y_input, pos_ids, neg_ids
	else:
		return X1_input, X2_input, node1_mask, node2_mask, y_input



def get_loss_epoch(model, graphs, classes, batch_size, load_data=None):
	if load_data is None:
		epoch_data  =  generate_epoch_pair(graphs, classes, batch_size)
	else:
		epoch_data = load_data

	tot_loss = 0
	for cur_data in epoch_data:
		X1, X2, mask1, mask2, y = cur_data
		cur_loss = model.calc_loss(X1, X2, mask1, mask2, y)
		tot_loss += cur_loss

	return tot_loss / len(epoch_data)




def train_epoch(model, graphs, classes, batch_size, added_pairs = None, load_data=None):
	if load_data is None:
		epoch_data = generate_epoch_pair(graphs, classes, batch_size)
	else:
		epoch_data = load_data
	epoch_data += added_pairs
	perm = np.random.permutation(len(epoch_data))   #Random shuffle

	for index in perm:
		cur_data = epoch_data[index]
		X1, X2, mask1, mask2, y = cur_data
		model.train(X1, X2, mask1, mask2, y)



def get_auc_epoch(model, graphs, classes, batch_size, load_data=None, title="", output_str=""):
	tot_diff = []
	tot_truth = []

	if load_data is None:
		epoch_data= generate_epoch_pair(graphs, classes, batch_size)
	else:
		epoch_data = load_data


	for cur_data in epoch_data:
		X1, X2, m1, m2,y  = cur_data
		diff = model.calc_diff(X1, X2, m1, m2)
	#	print diff

		tot_diff += list(diff)
		tot_truth += list(y > 0)


	diff = np.array(tot_diff)
	truth = np.array(tot_truth)

	fpr, tpr, thres = roc_curve(np.array(truth), (1-np.array(diff))/2)
	np.savez(title+output_str+"roc.npz", fpr=fpr, tpr=tpr,thres=thres)
	model_auc = auc(fpr, tpr)

	return model_auc


def show_stat(Gs):
	i586 = 0
	arm = 0
	mips = 0
	for g in Gs:
		if '-i586-linux-O' in g.name:
			i586 += 1
		elif '-armeb-linux-O' in g.name:
			arm += 1
		elif 'mips-linux-O' in g.name:
			mips += 1

	print 'x86:{}'.format(i586)
	print 'arm:{}'.format(arm)
	print 'mips:{}'.format(mips)
