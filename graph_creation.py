import json
import networkx as nx
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from collections import Counter

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

def load_data():
    """Load the cleaned nodes and edges data"""
    try:
        with open("clean_nodes.json", "r", encoding="utf-8") as f:
            nodes = json.load(f)
        
        with open("clean_edges.json", "r", encoding="utf-8") as f:
            edges = json.load(f)
            
        print(f"✅ Loaded {len(nodes)} nodes and {len(edges)} edges")
        return nodes, edges
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure clean_nodes.json and clean_edges.json are in the current directory")
        return None, None

def create_networkx_graph(nodes, edges):
    """Create a NetworkX graph from the data"""
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in nodes:
        G.add_node(node["id"], **node)
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'])
    
    return G

def basic_graph_visualization(G, title="Network Graph"):
    """Create a basic network visualization"""
    plt.figure(figsize=(15, 12))
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='lightgray', 
                          alpha=0.6, 
                          width=0.8)
    
    # Draw nodes
    node_colors = plt.cm.viridis(np.linspace(0, 1, len(G.nodes())))
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=300,
                          alpha=0.8)
    
    # Add labels
    labels = {node: G.nodes[node]['name'][:15] + ('...' if len(G.nodes[node]['name']) > 15 else '') 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def network_statistics(G):
    """Calculate and display network statistics"""
    print("\n" + "="*60)
    print("📊 NETWORK STATISTICS")
    print("="*60)
    
    # Basic stats
    print(f"🔗 Number of nodes: {G.number_of_nodes()}")
    print(f"🔗 Number of edges: {G.number_of_edges()}")
    print(f"🔗 Network density: {nx.density(G):.4f}")
    print(f"🔗 Is connected: {nx.is_connected(G)}")
    
    if nx.is_connected(G):
        print(f"🔗 Average shortest path length: {nx.average_shortest_path_length(G):.4f}")
        print(f"🔗 Network diameter: {nx.diameter(G)}")
    else:
        print(f"🔗 Number of connected components: {nx.number_connected_components(G)}")
    
    print(f"🔗 Average clustering coefficient: {nx.average_clustering(G):.4f}")
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    print(f"\n📈 DEGREE STATISTICS")
    print(f"   • Average degree: {np.mean(degrees):.2f}")
    print(f"   • Median degree: {np.median(degrees):.2f}")
    print(f"   • Max degree: {max(degrees)}")
    print(f"   • Min degree: {min(degrees)}")
    
    # Top nodes by degree
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n🌟 TOP 10 MOST CONNECTED NODES:")
    for i, (node, degree) in enumerate(top_nodes, 1):
        name = G.nodes[node]['name']
        print(f"   {i:2d}. {name[:30]:<30} (degree: {degree})")
