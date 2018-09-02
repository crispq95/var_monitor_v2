try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

import matplotlib 
matplotlib.use('Agg')

import cProfile 
import os 
import pstats
import io
import marshal
import tempfile
import argparse
import numpy as np
import matplotlib.gridspec as gridspec

from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from matplotlib.backends.backend_pdf import PdfPages

import pprint 
import matplotlib.pyplot as plt
import networkx as nx 

ATTR_LIST = ['num_calls','nonrec_calls','tottime','cumtime']

FORMAT_DICT={'filename':0, 'line':1, 'name':2}

class Stat:
	def __init__(self):
		self._filename = ""
		self._line = -1
		self._name = ""

		self._num_calls = -1
		self._num_nonrec_calls = -1
		self._tottime = -1
		self._cumtime = -1
		self._subcalls = {}

	def __getattribute__(self, name):
		return object.__getattribute__(self, name)

	def set_num_calls(self, a):
		self._num_calls = a

	def set_num_nonrec_calls(self, a):
		self._num_nonrec_calls = a

	def set_tottime(self, a):
		self._tottime = a

	def set_cumtime(self, a):
		self._cumtime = a

	def set_subcalls(self, a):
		self._subcalls = a

	def set_file(self, filename, line, name):
		self._filename = filename
		self._line = line 
		self._name = name

	@property
	def filename(self):
		return self._filename
	
	@property
	def line(self):
		return self._line
	
	@property
	def name(self):
		return self._name
	

	@property
	def num_calls(self):
		return self._num_calls
	
	@property
	def num_rec_calls(self):
		return self._num_rec_calls
	
	@property
	def tottime(self):
		return self._tottime
	
	@property
	def cumtime(self):
		return self._cumtime

	@property
	def subcalls(self):
		return self._subcalls
	

def order_by(attr,stats_list):
	for i in range(1,len(stats_list)):
		for j in range(0, len(stats_list)-i):
			if(getattr(stats_list[j], attr) < getattr(stats_list[j+1], attr)):
				k = stats_list[j+1]
				stats_list[j+1] = stats_list[j]
				stats_list[j] = k

	return stats_list

def parse_args():
    parser = argparse.ArgumentParser(description="Time profiler")

    parser.add_argument("--data", "-d", required=False, help="cProfile output file")

    args = parser.parse_args()
    return args

def load_data(data_file): 
	statsfile = tempfile.NamedTemporaryFile()

	s = open(data_file,'r+b')
	statsfile.file = s 
	statsfile.name = data_file

	p2 = pstats.Stats(statsfile.name)
	#p2.sort_stats('tottime').print_stats(15)
	#p2.sort_stats('cumtime').print_stats(15)


	""" raw_stats stores cProfile data with the following format:
		[(filename1,line_num,name_of_method):[statistics:num_calls, num_nonrecursive_calls, tottime, cumtime, subcall_stats], ... ]
		where subcall_stats is a dict with the same format as the shown above. 
	"""
	raw_stats = marshal.load(statsfile.file)
	stats = []
	
	#print("DATA LEN : ", len(raw_stats))

	"""
	print("")

	print(raw_stats)

	print("")
	print("")
	print("")
	"""

	tree = {}

	for k in raw_stats:
		
		#stats_list.append(Stat(k,stats[k])) 
		for calles in raw_stats[k][4]:
			if calles[2].replace("'","") in tree.keys(): 
				tree[calles[2].replace("'","")].append(k[2].replace("'",""))

			else : 
				tree[calles[2].replace("'","")] = [k[2].replace("'","")]
			#print( k[2].replace("'","") ,' <- ' ,calles[2].replace("'",""))
			#childs.append(calles[2].replace("'",""))

		#tree[k[2].replace("'","")] = childs

		if raw_stats[k][3] > 0.5 :
			"""
			test = {}

			test['num_calls'] = raw_stats[k][0]
			test['num_nonrec_calls'] = raw_stats[k][1]
			test['tottime'] = raw_stats[k][2]
			test['cumtime'] = raw_stats[k][3]
			test['subcalls'] = raw_stats[k][4]
			"""
			test = Stat()			#We use the class Stat to make the data easy to read. 
			
			test.set_num_calls(raw_stats[k][0]) # num calls
			test.set_num_nonrec_calls(raw_stats[k][1]) # num non rec calls
			test.set_tottime(raw_stats[k][2]) # tottime
			test.set_cumtime(raw_stats[k][3])	# cumtime 
			test.set_subcalls(raw_stats[k][4])	# subcalls stats 

			test.set_file(k[0].replace("'",""),int(k[1]), k[2].replace("'",""))
			stats.append(test)

		#if k[2].replace("'","") == 'main': 
		#	print ("*) ", raw_stats[k][4])
			

	#for k,val in tree.items(): 
	#	print ("Parent (",k,") -> ", val)


	orderb = '_cumtime'
	stats = order_by(orderb, stats)
	
	functions = []
	for s in stats:
		if '<' not in s.name :
			functions.append(s.name)

	return stats,tree,functions

def plot_time(stats,gs, all_times=False):
	#plot : 
	#fig, ax = plt.subplots()
	#fig = plt.figure(figsize=(10, 15))	#,tight_layout=True
	#gs = gridspec.GridSpec(4, 2, hspace=0.05, top=0.94)
	ax_indx = 0

	if all_times : 
		ax = plt.subplot(gs[1,:])
		ax.set_title('Total time of all functions.')
	else:	
		ax = plt.subplot(gs[0,:])
		ax.set_title('Total time of script functions.')

	names = []
	cumtime = []
	ncalls = []
	tottime = []


	for s in stats:
		if not all_times : 
			if '<' not in s.name : 
				names.append(s.name)	
				cumtime.append(s.cumtime)
				ncalls.append(s.num_calls)
				tottime.append(s.tottime)
		else : 
			names.append(s.name)	
			cumtime.append(s.cumtime)
			ncalls.append(s.num_calls)
			tottime.append(s.tottime)



	y_pos = np.arange(len(names))
	cumtime.reverse()
	names.reverse()
	ncalls.reverse()
	tottime.reverse()

	ax.barh(y_pos, cumtime, align='center',color='teal')
	ax.barh(y_pos, tottime, align='center',color='paleturquoise')
	#ax.plot(ncalls, y_pos, '-o', color='mediumspringgreen')

	ax.set_yticks(y_pos)
	ax.set_yticklabels(names)
	ax.set_xlabel('Time (s)')

	#plt.show()


def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, 
                  pos = None, parent = None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.'''

    if pos == None:
        pos = {root:(xcenter,vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = list(G.neighbors(root)) 
    if parent != None:   #this should be removed for directed graphs.
        neighbors.remove(parent)  #if directed, then parent not in neighbors.
    if len(neighbors)!=0:
        dx = width/len(neighbors) 
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G,neighbor, width = dx, vert_gap = vert_gap, 
                                vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos, 
                                parent = root)
    return pos

def draw_graph(childs,functions,gs):
	print ("GRAPH ")
	root = 'main'
	plt.subplot(gs[3,:])

	final_graph = {}
	i = 0
	gtnodes = {}
	children = {}

	lista_principal = {}
	dict_funct = {}

	i = 0
	for f in childs.keys(): 
		lista_principal[f] = i 
		dict_funct[f] = [i]
		i+=1 


	#lista_principal['main'] = ['acc_mult','arr_mult', 'sleep_funcs']



	lista_secundaria = {}
	repes = []
	#dict_funct = lista_principal.copy()

	i = len(lista_principal)


	#	k = nombre funcion ||	val = identificador numerico
	for k,val in lista_principal.items(): 
		belongs = {}
		children = []

		#	para cada hijo del la funcion k 
		for c in childs[k]: 
			if c in lista_principal.keys():
				if c not in repes: 
					children.append(lista_principal[c])
					repes.append(c)
				else: 
					#print ("ESTA REPE Y DENTRO DE LA LISTA -- ", c)
					if c in belongs: 
						belongs[c].append(i)
					else: 
						belongs[c] = [i]
					i+=1 
			elif c in repes:
				#print(c, ' repe !')
				if c in belongs.keys():
					belongs[c].append(i)

				else : 
					belongs[c] = [i]
				i+=1 

			else:  
				#print(c, 'NO repe !')
				repes.append(c)

				belongs[c] = [i]
				i+=1 

		"""
		print("BELONGS : ", k, ' -- ', belongs)
		print("CHILDREN : ", k, ' -- ', children)
		print ("")
		"""

		for b,bval in belongs.items(): 
			#print ('key: ', b, ' vals -- ', bval)
			for v in bval:
				#print("val -> ", v)
				children.append(v)
				if b in dict_funct.keys(): 
					dict_funct[b].append(v)
				else: 
					dict_funct[b] = [v]

		lista_secundaria[val] = children

	"""
	G = nx.Graph()


	for k,val in lista_secundaria.items(): 
		#parent = test_cutre(dict_funct, k)
		for child in val:
			#c = test_cutre(dict_funct, child)
			G.add_edge(k,child)
		print ("")

	
	pos = hierarchy_pos(G, lista_principal['main'])

	labels = {}
	for k,val in dict_funct.items():
		for v in val : 
			labels[v] = k

	print (labels)
	
	#nx.draw_networkx_labels(G, pos, labels, font_size=8)
	#nx.draw(G, pos=pos, node_color='white', node_shape='o', node_size=900)

	#pos = graphviz_layout(G)

	#A.layout(prog='dot')
	nx.draw(G, pos)
	
	A = nx.nx_agraph.to_agraph(G)
	pos = nx.nx_pydot.graphviz_layout(G)
	A.layout(G)
	A.draw('file.png')
	
	plt.show()
	plt.savefig("Graph.png", format="PNG")
	"""
	G = nx.DiGraph()
	
	G.add_node(lista_principal['main'])

	for k,val in lista_secundaria.items(): 
		for child in val:
			G.add_edge(k,child)
	write_dot(G,'test.dot')

	labels = {}
	for k,val in dict_funct.items():
		for v in val : 
			labels[v] = k

	plt.title('Function Tree')
	pos =graphviz_layout(G, prog='dot')
	#nx.draw_networkx_labels(G, pos, font_size=8)
	nx.draw(G, pos, with_labels=True, arrows=True)
	#plt.savefig('nodes.png')
	return dict_funct
	


def main(): 
	args = parse_args()
	childs = {}

	if args.data :
		data_file = args.data 
	else:  
		data_file = 'profile_output'
		

	ordered_stats,childs,functions = load_data(data_file)


	with PdfPages('test.pdf') as pdf:
		fig = plt.figure(figsize=(10, 15))  #,tight_layout=True
		gs = gridspec.GridSpec(4, 2)
		
		plot_time(ordered_stats,gs)
		plot_time(ordered_stats,gs,all_times=True)

		print('OS NAMES')
		txt1 = ''
		for s in ordered_stats: 
			txt1 += s.name+' (numcalls:'+str(s.num_calls)+') CALLED BY : '
			for sub in s.subcalls:
				txt1 += str(sub[2])+' ----> line =  '+str(sub[1])+' , '
			txt1 += '---/ \n'

		fig.text(0.07,0.3,txt1)

		plt.tight_layout()

		df = draw_graph(childs,functions,gs)

		txt2 = ''
		for k,v in df.items() : 
			txt2 += k +' -> '+str(v).replace('[','').replace(']','')+'\n'

		print('TEXT2')
		print(txt2)

		fig.text(0.07,0.1,txt2)

		pdf.savefig()
		plt.close(fig)

if __name__ == '__main__':
	main()
