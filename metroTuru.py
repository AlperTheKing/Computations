import networkx as nx

def find_all_eulerian_circuits(graph, start):
    circuits = []

    # Helper function for recursive backtracking
    def backtrack(path, edges_left):
        # If no edges are left, we've found a circuit
        if not edges_left:
            circuits.append(list(path))  # Record the circuit
            return
        
        current_node = path[-1]
        
        # Explore all neighbors
        for neighbor in list(graph.neighbors(current_node)):
            edge = (current_node, neighbor)
            reverse_edge = (neighbor, current_node)
            
            # If the edge exists in the remaining edges, use it
            if edge in edges_left or reverse_edge in edges_left:
                # Use the edge (current_node, neighbor)
                path.append(neighbor)
                # Remove both directions of the edge (undirected)
                edges_left.discard(edge)  
                edges_left.discard(reverse_edge)  
                
                # Continue the search
                backtrack(path, edges_left)
                
                # Backtrack: restore the edge and path
                path.pop()
                edges_left.add(edge)
                edges_left.add(reverse_edge)
    
    # Convert graph edges into a set for easy tracking
    edge_set = set(graph.edges())
    
    # Start backtracking from the starting vertex
    backtrack([start], edge_set)
    
    return circuits

# Create the graph
G = nx.Graph()
edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'E'), ('B', 'D'),
         ('C', 'E'), ('C', 'F'), ('D', 'E'), ('E', 'F')]

G.add_edges_from(edges)

# Find and print all Eulerian circuits
all_circuits = find_all_eulerian_circuits(G, 'A')

# Print the results
for i, circuit in enumerate(all_circuits, 1):
    print(f"Circuit {i}: {' -> '.join(circuit)}")
