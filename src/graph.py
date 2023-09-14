"""
Module to build Graph structures from a dataframe, either:
  - NetworkX graph (G)
  - HeteroData PyG graph
"""
import torch
from torch_geometric.data import HeteroData
import networkx as nx
# docker test
# Df features used in each flow node

flow_features = [
    " Flow Duration",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Fwd IAT Mean",
    " Bwd IAT Mean",
]


def get_flow_features():
    """
    flow 구성을 위한 features 리스트 retrun 

    Returns:
        - Flow features list
    """
    return flow_features


def create_labels_encode_map(df):
    """
    데이터 프레임에서
    공격분류를 index 숫자와 딕셔너리 형태로 리턴
    ex)    
        A공격 => 0
        B공격 => 1

    Args:
        - df
    Returns:
        - Dictionary containing { Label: label_id }
    """
    
    labels = df[" Label"].unique()

    ids = range(len(labels))
    return dict(zip(labels, ids))


def create_hosts_ip_nodes(df):
    """
    
    Source IP 와 Destination IP Unique number를 하나씩 부여하고 딕셔너리로 return 하는 함수 

    Args:
        - df
    Returns:
        - Dictionary containing a map of { host_ip: host_id (row id in hosts tensor)}
    """
    unique_hosts = set()
    hosts_map = {}

    id = 0

    # Iterate each row and look into Src and Dst hosts.
    for _, row in df.iterrows():
        src_host = row[" Source IP"]
        if src_host not in unique_hosts:
            hosts_map[src_host] = id
            id = id + 1
            unique_hosts.add(src_host)
        dst_host = row[" Destination IP"]
        if dst_host not in unique_hosts:
            hosts_map[dst_host] = id
            id = id + 1
            unique_hosts.add(dst_host)
    return hosts_map


def create_hosts_ip_port_nodes(df):
    """

    IP와 Port 튜플을 하나의 Node 노드로 구성하여
    Unique id를 부여하여 딕셔너리 return 

    Args:
        - df
    Returns:
        - Dictionary containing a map of { host_ip_port: host_id (row id in hosts tensor)}
    """
    unique_hosts = set()
    hosts_map = {}
    id = 0
    # Iterate each row and look into Src and Dst hosts.
    for _, row in df.iterrows():
        src_host = (row[" Source IP"], row[" Source Port"])
        if src_host not in unique_hosts:
            hosts_map[src_host] = id
            id = id + 1
            unique_hosts.add(src_host)
        dst_host = (row[" Destination IP"], row[" Destination Port"])

        if dst_host not in unique_hosts:
            hosts_map[dst_host] = id
            id = id + 1
            unique_hosts.add(dst_host)
    return hosts_map


def get_feature(row, feature_name):
    """
    inf, nan 등을 0으로 대치하는 함수
    그 외에는 기본 값 유지 

    Args:
        - row: dataframe row
        - feature_name
    Returns:
        - feature value
    """
    feature_value = row[feature_name]
    try:
        feature_value = float(feature_value)
        if feature_value != float('+inf') and feature_value != float('nan'):
            return feature_value
        else:
            return 0
    except:
        return 0


def get_flow_features_values(row, flow_features):
    """
    Args:
        - row: dataframe row
        - flow_features: list of features to include in each flow
    Returns:
        - list of feature values
    """
    flow_features_values = []
    for feature_name in flow_features:
        flow_features_values.append(get_feature(row, feature_name)) # inf, nan은 0으로 대치하고 나머지는 다시 value로 넣음

    return flow_features_values 


def get_encoded_label(row, labels_encode_map):
    """
    Args:
        -row: dataframe row
        - labels_encode_map: mapping of label name to label identifier (integer)

    ex) row = 'A공격' , labels_encode_map = {A공격 :0, B공격 :1, ...}
      ==> result  : 0

    Returns:
        - label identifier
    """
    return labels_encode_map[row[" Label"]]


def create_flow_nodes(df):
    """
    unique 한 Flow ID 마다 x,y를 만들고 ,
    flow_id : index number를 return 하는 함수 

    Args:
        - df
    Returns:
        - flows_map: mapping of {Flow: flow identifier}. Used to map each flow to the features tensor row.
        - x: flow features tensor -> [N of flows, Features per flow].
        - y: flows labels tensor -> [N of flows] (each position contains the encoded label associated to each flow)
    """
    labels_encode_map = create_labels_encode_map(df)   # {Target class : index} 꼴의 딕셔너리
    unique_flows = set()
    flows_map = {}
    i = 0
    x, y = [], []

    for _, row in df.iterrows():
        flow_id = row["Flow ID"]
        if flow_id not in unique_flows:
            flows_map[flow_id] = i
            i = i + 1
            unique_flows.add(flow_id)
            x.append(get_flow_features_values(row, flow_features)) # feature value 형태로 만들어 x에 넣음 
            y.append(get_encoded_label(row, labels_encode_map))    # target의 class를 index 숫자로 대치하여 넣음 

    return torch.FloatTensor(x), torch.LongTensor(y), flows_map  #Featue value  /  Target  / {flow id : unique number}




def create_edge_index(df, hosts_ip_map, hosts_ip_port_map, flows_map):
    """
    This method creates edge indexes for the 4 different type of edges we can have in the graph:
        - IP node -> IP/Port Node
        - IP node <- IP/Port Node
        - IP/Port node -> Connection node
        - IP/Port node <- Connection node
    (see repo graphs to better understand edges and node types)

    Args:
        - df
        - hosts_ip_map: map containing host IP nodes ids.
        - hosts_ip_port_map: map containing host IP/Port ids.
        - Flows_map: map containing flows ids.

    Returns:
        - 4 Long Tensors corresponsing to each of the edge indexes (COO format)
    """

    src0, dst0 = [], []
    src1, dst1 = [], []
    src2, dst2 = [], []
    src3, dst3 = [], []

    for _, row in df.iterrows():

        src_ip = row[" Source IP"]
        src_tup = (row[" Source IP"], row[" Source Port"])
        flow = row["Flow ID"]
        dst_tup = (row[" Destination IP"], row[" Destination Port"])
        dst_ip = row[" Destination IP"]

        # Edge: Ip Node -> Ip/Port Node
        src0.append(hosts_ip_map[src_ip])
        dst0.append(hosts_ip_port_map[src_tup])
        src0.append(hosts_ip_map[dst_ip])
        dst0.append(hosts_ip_port_map[dst_tup])

        # Edge: Ip/Port Node -> Connection Node
        src1.append(hosts_ip_port_map[src_tup])
        dst1.append(flows_map[flow])
        src1.append(hosts_ip_port_map[dst_tup])
        dst1.append(flows_map[flow])

        # Edge: Connection Node -> Ip/Port Node
        src2.append(flows_map[flow])
        dst2.append(hosts_ip_port_map[dst_tup])
        src2.append(flows_map[flow])
        dst2.append(hosts_ip_port_map[src_tup])

        # Edge: Ip/Port Node -> Node
        src3.append(hosts_ip_port_map[dst_tup])
        dst3.append(hosts_ip_map[dst_ip])
        src3.append(hosts_ip_port_map[src_tup])
        dst3.append(hosts_ip_map[src_ip])

    return torch.LongTensor([src0, dst0]), torch.LongTensor([src1, dst1]), torch.LongTensor([src2, dst2]), torch.LongTensor([src3, dst3])


def get_hosts_tensor(hidden_dimension, hosts_map):
    """
    Args:
        - hidden dimension
        - hosts_map: mapping of {Host: host id (row id in hosts features tensor)}
    Returns:
        - tensor initialzied to 1s [N of hosts, Hidden dimension]
    """
    return torch.ones(len(hosts_map.keys()), hidden_dimension)


def create_hetero_graph(df):
    """
    Args:
        - df
    Returns:
        - HeteroData object containing edges and nodes (host and connection types)
    """
    hosts_ip = create_hosts_ip_nodes(df)  # ip : unique number 딕셔너리 생성 
    hosts_ip_port = create_hosts_ip_port_nodes(df) # (ip,port) : unique number 딕셔너리 생성 

    x_hosts_ip = get_hosts_tensor(128, hosts_ip) # host ip 기준으로 tensor 생성
    x_hosts_ip_port = get_hosts_tensor(128, hosts_ip_port) # Ip와 Port 기준으로 tensor 생성 

    x_flows, y_flows, flows_map = create_flow_nodes(df)  # X , Y , {flow id : unique num }

    ip_to_port, port_to_flow, flow_to_port, port_to_ip = create_edge_index(
        df, hosts_ip, hosts_ip_port, flows_map)

    hetero_data = HeteroData()

    # We add three different types of nodes
    hetero_data["host_ip_port"].x = x_hosts_ip_port
    hetero_data["host_ip"].x = x_hosts_ip
    hetero_data["connection"].x = x_flows
    hetero_data["connection"].y = y_flows

    # We add four different types of edges
    hetero_data["host_ip", "to", "host_ip_port"].edge_index = ip_to_port
    hetero_data["host_ip_port", "to", "connection"].edge_index = port_to_flow
    hetero_data["connection", "to", "host_ip_port"].edge_index = flow_to_port
    hetero_data["host_ip_port", "to", "host_ip"].edge_index = port_to_ip

    return hetero_data


def create_networkx_graph(df, flow_labels):
    """
    Args:
        - df
        - flow_labels: list containing mapping of flow_id to label predicted (for coloring purposes)
    Returns:
        - G: networkX graph
        - colors: colors list (len == number of nodes in G)
    """
    G = nx.DiGraph()
    hosts_ip = create_hosts_ip_nodes(df)
    hosts_ip_port = create_hosts_ip_port_nodes(df)
    _, _, flows_map = create_flow_nodes(df)
    colors = []
    for ip in hosts_ip.keys():
        G.add_node(ip)
        colors.append("blue")

    for port in hosts_ip_port.keys():
        G.add_node(port)
        colors.append("blue")

    for flow in flows_map.keys():
        if flow_labels[flows_map[flow]] == 0:
            colors.append("green")
        else:
            colors.append("red")

    for _, row in df.iterrows():
        src0 = row[" Source IP"]
        dst0 = (row[" Source IP"], row[" Source Port"])
        src1 = dst0
        dst1 = row["Flow ID"]
        src2 = dst1
        dst2 = (row[" Destination IP"], row[" Destination Port"])
        src3 = dst2
        dst3 = row[" Destination IP"]
        G.add_edge(src0, dst0)
        G.add_edge(src1, dst1)
        G.add_edge(src2, dst2)
        G.add_edge(src3, dst3)

    return G, colors
