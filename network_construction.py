"""
Step 1: Network Construction
Creates multilayer simplicial complex network
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
import json

class MultiLayerSimplicialNetwork:
    def __init__(self, N=1000, k1=10, k2=2, er_prob=0.006):
        """
        N: Number of nodes
        k1: Average degree (1-simplices per node in info layer)
        k2: Number of 2-simplices per node
        er_prob: ER graph connection probability for physical layer
        """
        self.N = N
        self.k1 = k1
        self.k2 = k2
        self.er_prob = er_prob
        
        # Networks
        self.physical_layer = None 
        self.info_layer_graph = None  # awareness layer
        self.simplices_2 = []  # list of 2-simplices 
        
        print("="*20)
        print("Building Multilayer Network")
        print("="*20)
        
        self._build_physical_layer()
        self._build_information_layer()
        self._compute_statistics()
        
    def _build_physical_layer(self):
        """Build ER random graph for disease transmission"""
        print(f"\n===== [1/2] Building Physical Layer =====")
        print(f"  Erdos-Renyi random graph")
        print(f"  Connection probability: {self.er_prob}")
        
        self.physical_layer = nx.erdos_renyi_graph(self.N, self.er_prob)
        
        num_edges = self.physical_layer.number_of_edges()
        avg_degree = 2 * num_edges / self.N
        
        print(f"\n  Physical Layer Created:")
        print(f"    - Nodes: {self.N}")
        print(f"    - Edges: {num_edges}")
        print(f"    - Average degree: {avg_degree:.2f}")
        
    def _build_information_layer(self):
        """Build simplicial complex for awareness layer"""
        print(f"\n===== [2/2] Building Information Layer =====")
        print(f"  Type: Random simplicial complex")
        print(f"  Target k1 (avg degree): {self.k1}")
        print(f"  Target k2 (2-simplices per node): {self.k2}")
        
        # Calculate probabilities from Fan et al. Eq (2.1)
        p2 = (2 * self.k2) / ((self.N - 1) * (self.N - 2))
        p1 = (self.k1 - 2*self.k2) / ((self.N - 1) - 2*self.k2)
        
        print(f"\n  Calculated probabilities:")
        print(f"    - p1 (1-simplex): {p1:.6f}")
        print(f"    - p2 (2-simplex): {p2:.8f}")
        
        # Step 1: Generate base ER graph for 1-simplices (edges)
        print(f"\n  Step 1: Creating base graph (1-simplices)...")
        self.info_layer_graph = nx.erdos_renyi_graph(self.N, p1)
        base_edges = self.info_layer_graph.number_of_edges()
        print(f"    - Base edges created: {base_edges}")
        
        # Step 2: Add 2-simplices (triangles)
        print(f"\n  Step 2: Adding 2-simplices (triangles)...")
        self.simplices_2 = set()  # Use set to avoid duplicates
        
        for i in range(self.N):
            # Try to form k2 triangles for each node
            candidates = [n for n in range(self.N) if n != i]
            
            for j, k in combinations(candidates, 2):
                if np.random.random() < p2:
                    # Create 2-simplex (triangle)
                    simplex = tuple(sorted([i, j, k]))
                    
                    if simplex not in self.simplices_2:
                        # Add all three edges
                        self.info_layer_graph.add_edge(i, j)
                        self.info_layer_graph.add_edge(i, k)
                        self.info_layer_graph.add_edge(j, k)
                        
                        self.simplices_2.add(simplex)
        
        self.simplices_2 = list(self.simplices_2)
        
        final_edges = self.info_layer_graph.number_of_edges()
        final_avg_degree = 2 * final_edges / self.N
        simplices_per_node = 3 * len(self.simplices_2) / self.N
        
        print(f"\n   Information Layer Created:")
        print(f"     Total edges (1-simplices): {final_edges}")
        print(f"     Total triangles (2-simplices): {len(self.simplices_2)}")
        print(f"     Actual average degree: {final_avg_degree:.2f} (target: {self.k1})")
        print(f"     Actual 2-simplices per node: {simplices_per_node:.2f} (target: {self.k2})")
        
    def _compute_statistics(self):
        """Compute and display network statistics"""
        print(f"\n" + "="*20)
        print("Network Stats")
        print("="*20)
        
        # Physical layer stats
        phys_degree_seq = [d for n, d in self.physical_layer.degree()]
        phys_avg_degree = np.mean(phys_degree_seq)
        phys_std_degree = np.std(phys_degree_seq)
        phys_clustering = nx.average_clustering(self.physical_layer)
        
        print(f"\nPhysical Layer (Disease Network):")
        print(f"  Average degree: {phys_avg_degree:.2f} ± {phys_std_degree:.2f}")
        print(f"  Max degree: {max(phys_degree_seq)}")
        print(f"  Min degree: {min(phys_degree_seq)}")
        print(f"  Clustering coefficient: {phys_clustering:.4f}")
        
        # Information layer stats
        info_degree_seq = [d for n, d in self.info_layer_graph.degree()]
        info_avg_degree = np.mean(info_degree_seq)
        info_std_degree = np.std(info_degree_seq)
        info_clustering = nx.average_clustering(self.info_layer_graph)
        
        print(f"\nInformation Layer (Awareness Network):")
        print(f"  Average degree: {info_avg_degree:.2f} ± {info_std_degree:.2f}")
        print(f"  Max degree: {max(info_degree_seq)}")
        print(f"  Min degree: {min(info_degree_seq)}")
        print(f"  Clustering coefficient: {info_clustering:.4f}")
        print(f"  Number of 2-simplices: {len(self.simplices_2)}")
        
        # Analyze 2-simplex distribution
        simplex_counts = np.zeros(self.N)
        for simplex in self.simplices_2:
            for node in simplex:
                simplex_counts[node] += 1
        
        print(f"\n2-Simplex Distribution:")
        print(f"  Average per node: {np.mean(simplex_counts):.2f}")
        print(f"  Std dev: {np.std(simplex_counts):.2f}")
        print(f"  Max: {int(np.max(simplex_counts))}")
        print(f"  Min: {int(np.min(simplex_counts))}")
        
    def build_adjacency_structures(self):
        """
        Build adjacency lists and matrices for simulation
        """
        print(f"\n" + "="*20)
        print("Building Adjacency Structures")
        print("="*20)
        
        # Adjacency matrices
        print(f"\n===== [1/3] Building adjacency matrices =====")
        self.matrix_physical = nx.to_numpy_array(self.physical_layer, dtype=int)
        self.matrix_info = nx.to_numpy_array(self.info_layer_graph, dtype=int)
        
        # Adjacency lists
        print(f"\n===== [2/3] Building adjacency lists =====")
        self.adjacency_list_physical = [list(self.physical_layer.neighbors(i)) for i in range(self.N)]
        self.adjacency_list_info = [list(self.info_layer_graph.neighbors(i)) for i in range(self.N)]
        
        # 2-simplex adjacency list (for each node, store pairs of other nodes in its triangles)
        print(f"\n===== [3/3] Building 2-simplex adjacency structure =====")
        self.adjacency_triangles_list = [[] for _ in range(self.N)]
        
        for simplex in self.simplices_2:
            i, j, k = simplex
            self.adjacency_triangles_list[i].append([j, k])
            self.adjacency_triangles_list[j].append([i, k])
            self.adjacency_triangles_list[k].append([i, j])
        
        print(f"  Adjacency structures complete")
        
    def save_network(self, filepath='networks/multilayer_network.pkl'):
        print(f"\n" + "="*20)
        print("Saving Network")
        print("="*20)
        
        network_data = {
            'N': self.N,
            'k1': self.k1,
            'k2': self.k2,
            'er_prob': self.er_prob,
            'physical_layer': self.physical_layer,
            'info_layer_graph': self.info_layer_graph,
            'simplices_2': self.simplices_2,
            'matrix_physical': self.matrix_physical,
            'matrix_info': self.matrix_info,
            'adjacency_list_physical': self.adjacency_list_physical,
            'adjacency_list_info': self.adjacency_list_info,
            'adjacency_triangles_list': self.adjacency_triangles_list
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(network_data, f)
        
        print(f"  Network saved to: {filepath}")
        
        # Also save summary as JSON
        summary = {
            'N': self.N,
            'k1': self.k1,
            'k2': self.k2,
            'er_prob': self.er_prob,
            'physical_edges': self.physical_layer.number_of_edges(),
            'info_edges': self.info_layer_graph.number_of_edges(),
            'num_2simplices': len(self.simplices_2)
        }
        
        with open(filepath.replace('.pkl', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Summary saved to: {filepath.replace('.pkl', '_summary.json')}")
        
    def visualize(self, save_path='results/network_visualization.png'):
        print(f"\n" + "="*20)
        print("Visualizing Network")
        print("="*20)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multilayer Simplicial Network Analysis', fontsize=16, fontweight='bold')
        
        # Choose subset of network to visualize for clarity
        sample_nodes = np.random.choice(self.N, size=min(240, self.N), replace=False)
        phys_subgraph = self.physical_layer.subgraph(sample_nodes)
        info_subgraph = self.info_layer_graph.subgraph(sample_nodes)
        
        # 1. Physical layer network
        print(f"[1/6] Drawing physical layer...")
        pos_phys = nx.spring_layout(phys_subgraph, seed=42)
        nx.draw_networkx(phys_subgraph, pos_phys, ax=axes[0,0], 
                        node_size=50, node_color='lightcoral',
                        edge_color='gray', alpha=0.6, with_labels=False)
        axes[0,0].set_title(f'Physical Layer (Disease)\n{len(sample_nodes)} nodes sample', 
                           fontweight='bold')
        axes[0,0].axis('off')
        
        # 2. Information layer network with 2-simplices highlighted
        print(f"[2/6] Drawing information layer with 2-simplices...")
        pos_info = nx.spring_layout(info_subgraph, seed=42)
        
        # Draw base graph
        nx.draw_networkx_edges(info_subgraph, pos_info, ax=axes[0,1],
                              edge_color='lightgray', alpha=0.3)
        
        # Highlight 2-simplices in sample
        for simplex in self.simplices_2:
            if all(n in sample_nodes for n in simplex):
                triangle_edges = list(combinations(simplex, 2))
                nx.draw_networkx_edges(info_subgraph, pos_info, ax=axes[0,1],
                                     edgelist=triangle_edges,
                                     edge_color='lightcoral', width=2, alpha=0.6)
        
        nx.draw_networkx_nodes(info_subgraph, pos_info, ax=axes[0,1],
                             node_size=50, node_color='lightblue')
        axes[0,1].set_title(f'Information Layer (Awareness)\n2-simplices in red', 
                           fontweight='bold')
        axes[0,1].axis('off')
        
        # 3. Degree distributions
        print(f"[3/6] Plotting degree distributions...")
        phys_degrees = [d for n, d in self.physical_layer.degree()]
        info_degrees = [d for n, d in self.info_layer_graph.degree()]
        
        axes[0,2].hist(phys_degrees, bins=30, alpha=0.6, label='Physical', color='red')
        axes[0,2].hist(info_degrees, bins=30, alpha=0.6, label='Information', color='blue')
        axes[0,2].axvline(np.mean(phys_degrees), color='red', linestyle='--', linewidth=2)
        axes[0,2].axvline(np.mean(info_degrees), color='blue', linestyle='--', linewidth=2)
        axes[0,2].set_xlabel('Degree')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Degree Distribution Comparison', fontweight='bold')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. 2-Simplex distribution
        print(f"[4/6] Plotting 2-simplex distribution...")
        simplex_counts = np.zeros(self.N)
        for simplex in self.simplices_2:
            for node in simplex:
                simplex_counts[node] += 1
        
        axes[1,0].hist(simplex_counts, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[1,0].axvline(np.mean(simplex_counts), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {np.mean(simplex_counts):.2f}')
        axes[1,0].set_xlabel('Number of 2-Simplices per Node')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('2-Simplex Distribution', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Clustering coefficient comparison
        print(f"[5/6] Computing clustering coefficients...")
        phys_clustering = list(nx.clustering(self.physical_layer).values())
        info_clustering = list(nx.clustering(self.info_layer_graph).values())
        
        axes[1,1].hist(phys_clustering, bins=30, alpha=0.6, label='Physical', color='red')
        axes[1,1].hist(info_clustering, bins=30, alpha=0.6, label='Information', color='blue')
        axes[1,1].set_xlabel('Clustering Coefficient')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Clustering Coefficient Distribution', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Network statistics table
        print(f"[6/6] Creating statistics table...")
        stats_data = [
            ['Property', 'Physical Layer', 'Information Layer'],
            ['Nodes', f'{self.N}', f'{self.N}'],
            ['Edges', f'{self.physical_layer.number_of_edges()}', 
             f'{self.info_layer_graph.number_of_edges()}'],
            ['Avg Degree', f'{np.mean(phys_degrees):.2f}', f'{np.mean(info_degrees):.2f}'],
            ['Avg Clustering', f'{np.mean(phys_clustering):.4f}', 
             f'{np.mean(info_clustering):.4f}'],
            ['2-Simplices', 'N/A', f'{len(self.simplices_2)}']
        ]
        
        axes[1,2].axis('tight')
        axes[1,2].axis('off')
        table = axes[1,2].table(cellText=stats_data, cellLoc='center', loc='center',
                               colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('green')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1,2].set_title('Network Statistics Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  Visualization saved to: {save_path}")
        plt.show()
        
    @staticmethod
    def load_network(filepath='networks/multilayer_network.pkl'):
        with open(filepath, 'rb') as f:
            network_data = pickle.load(f)
        
        # Reconstruct object
        network = MultiLayerSimplicialNetwork.__new__(MultiLayerSimplicialNetwork)
        network.N = network_data['N']
        network.k1 = network_data['k1']
        network.k2 = network_data['k2']
        network.er_prob = network_data['er_prob']
        network.physical_layer = network_data['physical_layer']
        network.info_layer_graph = network_data['info_layer_graph']
        network.simplices_2 = network_data['simplices_2']
        network.matrix_physical = network_data['matrix_physical']
        network.matrix_info = network_data['matrix_info']
        network.adjacency_list_physical = network_data['adjacency_list_physical']
        network.adjacency_list_info = network_data['adjacency_list_info']
        network.adjacency_triangles_list = network_data['adjacency_triangles_list']
        
        return network


def main():    
    network = MultiLayerSimplicialNetwork(
        N=1000,      # Population size
        k1=4,       # Average degree in information layer
        k2=1,        # 2-simplices per node
        er_prob=0.005  # ER probability for physical layer
    )
    
    # Build adjacency structures for simulation
    network.build_adjacency_structures()
    network.save_network()
    network.visualize()

if __name__ == "__main__":
    main()