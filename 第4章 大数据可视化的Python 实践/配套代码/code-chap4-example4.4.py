#%%
#创建一个没有节点和边的图，用不同的方法添加节点和边，并查看相关情况
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3)], color='cadetblue')
G.add_node("spam", time='2pm')
G.add_nodes_from("spam", time='5pm')
G.add_edge(3, 'm', weight=2)
new_edge = ('m','spam')
G.add_edge(*new_edge, weight=3.3)

nx.draw_shell(G, with_labels=True, node_size=1200, 
              node_color='cadetblue', font_size=15, font_color='k', font_weight='bold')
plt.savefig('networkx1.png', dpi=300, bbox_inches='tight')

#%%
#通过下标获取邻接信息和属性信息，并赋属性
G.nodes[1]['time'] = '3pm'
G[1][3]['weight'] = 4.7
G.edges[1, 2]['weight'] = 3.5

print ('adj of node 1:',G[1])
# output: 
# adj of node 1: {2: {'color': 'cadetblue', 'weight': 3.5}, 
#                 3: {'color': 'cadetblue', 'weight': 4.7}}
print ('attributes of edge 13:',G[1][3])
# output:
# attributes of edge 13: {'color': 'cadetblue', 'weight': 4.7}
print ('attributes of edge 12:',G.edges[1, 2])
# output:
# attributes of edge 12: {'color': 'cadetblue', 'weight': 3.5}

#%%
print ('--')
print ('num of nodes:',G.number_of_nodes())
# output:
# num of nodes: 8
print ('num of edges:',G.number_of_edges())
# output: 
# num of edges: 4
print ('node list:', list(G.nodes))
# output: 
# node list: [1, 2, 3, 'spam', 's', 'p', 'a', 'm']
print ('edge list:',list(G.edges))
# output: 
# edge list: [(1, 2), (1, 3), (3, 'm'), ('spam', 'm')]
print ('adj node list of 1:',list(G.adj[1]))
# output: adj node list of 1: [2, 3]
print ('edges of 1&m:',G.edges([1, 'm']))
# output: 
# edges of 1&m: [(1, 2), (1, 3), ('m', 3), ('m', 'spam')]
print ('edge degree of 1:',G.degree[1])
# output: 
# edge degree of 1: 2
print ('degree of 2&3:',G.degree([2, 3]))
# output: 
# degree of 2&3: [(2, 1), (3, 2)]
print ('--')

#%%
plt.subplot(111)
nx.draw_shell(G, with_labels=True, node_color='firebrick', alpha=0.8)

#%%
#删除边和节点，创建双向图
G.remove_node(2)
G.remove_nodes_from("spam")
print ('nodes list after removal:',list(G.nodes))
G.remove_edge(1, 3)
G.add_edge(1, 2)
H = nx.DiGraph(G)
print ('',list(H.edges()))

#%%
#快速检查所有的节点及其临近关系（在临近关系迭代中，无向图的所有边会出现两次）
FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
FG.add_node(0)
print (FG.adj)

print ('node, neighbour, weight(<0.5):')
for n, nbrs in FG.adj.items():
    for nbr, eattr in nbrs.items():
        wt = eattr['weight']
        if wt < 0.5: print('(%d, %d, %.3f)' % (n, nbr, wt))

# output:
# {1: {2: {'weight': 0.125}, 3: {'weight': 0.75}}, 
#  2: {1: {'weight': 0.125}, 4: {'weight': 1.2}}, 
#  3: {1: {'weight': 0.75}, 4: {'weight': 0.375}}, 
#  4: {2: {'weight': 1.2}, 3: {'weight': 0.375}}, 
#  0: {}}
# node, neighbour, weight(<0.5):
# (1, 2, 0.125)
# (2, 1, 0.125)
# (3, 4, 0.375)
# (4, 3, 0.375)

# 快速遍历的方法
#%%
print ('node, neighbour, weight(<0.5):')
for (u, v, wt) in FG.edges.data('weight'):
    if wt < 0.5: print('(%d, %d, %.3f)' % (u, v, wt))

#%%
#用图理论分析结构
print ('connected components:',list(nx.connected_components(FG)))
# output:
# connected components: [{1, 2, 3, 4}, {0}]
print ('sorted degree:', sorted(d for n, d in FG.degree()))
# output: 
# sorted degree: [0, 2, 2, 2, 2]
print ('bipartite clustering coefficient for nodes:')
print (nx.clustering(FG))
# output: 
# bipartite clustering coefficient for nodes:
# {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}

#%%
#获取最短路
print ('shortest path of 1&4:', nx.shortest_path(FG, 1,4))
# output:
# shortest path of 1&4: [1, 2, 4]
print ('shortest path of the network:')
print (dict(nx.all_pairs_shortest_path(FG)))
# output:
# shortest path of the network:
# {1: {1: [1], 2: [1, 2], 3: [1, 3], 4: [1, 2, 4]}, 
#  2: {2: [2], 1: [2, 1], 4: [2, 4], 3: [2, 1, 3]}, 
#  3: {3: [3], 1: [3, 1], 4: [3, 4], 2: [3, 1, 2]}, 
#  4: {4: [4], 2: [4, 2], 3: [4, 3], 1: [4, 2, 1]}, 
#  0: {0: [0]}}

#%%
#绘图及保存
import networkx as nx
import matplotlib.pyplot as plt
G = nx.petersen_graph()
options = {
    'node_color': 'cadetblue',
    'node_size': 450,
    'width': 1,
    'with_labels': True,
    'font_weight': 'bold',
    'font_size': 15,
}
plt.figure(figsize=(10,8), dpi=300)
plt.subplot(221)
nx.draw_random(G, **options)
plt.subplot(222)
nx.draw_circular(G, **options)
plt.subplot(223)
nx.draw_spectral(G, **options)
plt.subplot(224)
nx.draw_shell(G, nlist=[range(5,10), range(5)], **options)

plt.savefig('petersen_graph.png')

#%%
plt.subplot(111)
nx.draw(G, with_labels=True, node_color='firebrick', alpha=0.8, )

#%%
#有向图
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
print (DG.out_degree(1, weight='weight'))
print (DG.degree(1, weight='weight'))
print (list(DG.successors(1)))
print (list(DG.neighbors(1)))
#有向图和无向图的转换

#%%
#其他生成图的方法

petersen = nx.petersen_graph()
tutte = nx.tutte_graph()
maze = nx.sedgewick_maze_graph()
tet = nx.tetrahedral_graph()
K_5 = nx.complete_graph(5)
K_3_5 = nx.complete_bipartite_graph(3, 5)
barbell = nx.barbell_graph(10, 10)
lollipop = nx.lollipop_graph(10, 20)
#%%
plt.subplot(111)
nx.draw(K_3_5, with_labels=True, node_color='firebrick', alpha=0.8)

#%%
