import networkx as nx

# Define the function to validate Eulerian circuits
def validate_circuit(graph, circuit):
    # Convert the circuit to a list of edges
    circuit_edges = [(circuit[i], circuit[i + 1]) for i in range(len(circuit) - 1)]
    
    # Check if every edge in the graph is used exactly once
    graph_edges = set((min(edge), max(edge)) for edge in graph.edges())  # Handle undirected edges
    used_edges = set()

    for edge in circuit_edges:
        # Handle undirected edges by ensuring both (u, v) and (v, u) are considered the same
        edge_tuple = (min(edge), max(edge))
        if edge_tuple in used_edges:
            return False, "Edge used more than once"
        if edge_tuple not in graph_edges:
            return False, f"Edge {edge_tuple} does not exist in the graph"
        used_edges.add(edge_tuple)

    # Check if all edges are used exactly once
    if len(used_edges) != len(graph_edges):
        return False, "Not all edges used"

    # Check if the circuit starts and ends at the same vertex
    if circuit[0] != circuit[-1]:
        return False, "Does not start and end at the same vertex"

    return True, "Valid circuit"

# Create the graph
G = nx.Graph()
edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'E'), ('B', 'D'),
         ('C', 'E'), ('C', 'F'), ('E', 'D'), ('E', 'F')]  # Edge ('E', 'D') included as undirected
G.add_edges_from(edges)

# Print the edges of the graph
print("Edges in the graph:")
for edge in G.edges():
    print(edge)

# Circuits from your results (adding the first two for testing)
circuits = [
    ['A', 'B', 'C', 'E', 'B', 'D', 'E', 'F', 'C', 'A'],  # Circuit 1
    ['A', 'B', 'C', 'E', 'D', 'B', 'E', 'F', 'C', 'A'],  # Circuit 2
]

# Validate each circuit
for i, circuit in enumerate(circuits, 1):
    is_valid, message = validate_circuit(G, circuit)
    print(f"Circuit {i}: {message}")