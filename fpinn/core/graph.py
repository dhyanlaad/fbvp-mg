import networkx as nx

def build_graph(edges_def):
    G = nx.Graph()
    for idx, (u, v, length) in enumerate(edges_def):
        G.add_edge(u, v, edge_id=idx, length=length, start=u, end=v)
    return G

def get_edge_list(G):
    edges = []
    for _, _, data in G.edges(data=True):
        edges.append((data['edge_id'], data['start'], data['end'], data['length']))
    edges.sort(key=lambda e: e[0])
    return edges

def get_vertex_info(G):
    info = {}
    for v in G.nodes():
        incident = []
        for _, _, data in G.edges(v, data=True):
            edge_id = data['edge_id']
            if data['start'] == data['end']:
                incident.append((edge_id, True))
                incident.append((edge_id, False))
            else:
                is_start = (v == data['start'])
                incident.append((edge_id, is_start))
        info[v] = incident
    return info

def get_internal_vertices(G):
    return [v for v in G.nodes() if G.degree(v) > 1]

def get_boundary_vertices(G):
    return [v for v in G.nodes() if G.degree(v) == 1]
