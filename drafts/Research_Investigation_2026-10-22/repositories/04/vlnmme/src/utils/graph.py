from collections import defaultdict, deque

class NavGraph:
    def __init__(self):
        self.reset()

    def _register_node(self, viewpoint_id):
        """Assign a numeric node ID to a new viewpoint_id if not already assigned."""
        if viewpoint_id not in self.viewpoint2nodeid:
            node_id = f"node_{self.next_node_id}"
            self.viewpoint2nodeid[viewpoint_id] = node_id
            self.nodeid2viewpoint[node_id] = viewpoint_id
            self.next_node_id += 1
            self.all_nodes.add(node_id)
            return node_id

    def add_node(self, viewpoint_id):
        self._register_node(viewpoint_id)
        if viewpoint_id not in self.graph:
            self.graph[viewpoint_id] = []

    def update_connection(self, node1_vp, node2_vp):
        self._register_node(node1_vp)
        self._register_node(node2_vp)
        self.add_node(node1_vp)
        self.add_node(node2_vp)

        if node2_vp not in self.graph[node1_vp]:
            self.graph[node1_vp].append(node2_vp)
        if node1_vp not in self.graph[node2_vp]:
            self.graph[node2_vp].append(node1_vp)

        # Rebuild connection string
        self._update_connection_string()

    def _update_connection_string(self):
        lines = []
        # Sort by the integer part of the node ID for deterministic output
        sorted_vps = sorted(self.graph.keys(), key=lambda vp: int(self.viewpoint2nodeid[vp][5:]))
        for vp in sorted_vps:
            node_id = self.viewpoint2nodeid[vp]
            neighbor_ids = [self.viewpoint2nodeid[n] for n in self.graph[vp]]
            # Sort neighbors by the integer part of their node ID
            sorted_neighbor_ids = sorted(neighbor_ids, key=lambda nid: int(nid[5:]))
            neighbor_str = ", ".join(sorted_neighbor_ids)
            lines.append(f"{node_id} is connected to {neighbor_str}" if neighbor_str else f"{node_id} has no connections")
        self.connection = "\n".join(lines)

    def update_current_position(self, new_viewpoint_id):
        """Updates the agent's current position and marks it as visited."""
        numbered_id = self.viewpoint2nodeid[new_viewpoint_id]
        self.curr_viewpoint = numbered_id
        
        if len(self.trajectory) == 0 or self.trajectory[-1] != numbered_id:
            self.trajectory.append(numbered_id)

        # --- New addition: mark the new viewpoint as visited ---
        self.visited_nodes.add(numbered_id)
        
        # --- New addition: update the visit status string ---
        self._update_visit_status_string()

    def _update_visit_status_string(self):
        """
        Updates a string showing:
        - the current viewpoint,
        - all globally visited node IDs,
        - all globally unvisited node IDs.
        """
        parts = []

        # Show current viewpoint
        if self.curr_viewpoint is None:
            parts.append("Current viewpoint: None. Not in the graph.")
        else:
            summary = f"which is a {self.curr_summary}" if self.curr_summary else ""
            parts.append(f"Current viewpoint: {self.curr_viewpoint}. {summary}")

        # Compute and format visited/unvisited
        visited = sorted(self.visited_nodes, key=lambda nid: int(nid[5:]))
        unvisited = sorted(self.all_nodes - self.visited_nodes, key=lambda nid: int(nid[5:]))

        parts.append(f"Visited nodes: {', '.join(visited)}" if visited else "Visited nodes: None.")
        parts.append(f"Unvisited nodes : {', '.join(unvisited)}" if unvisited else "Unvisited nodes: None.")

        self.visit_status_string = " ".join(parts)


    # def _update_visit_status_string(self):
    #     """
    #     Generates a string describing the visited status of neighbors
    #     from the current viewpoint.
    #     """
    #     if self.curr_viewpoint is None:
    #         self.visit_status_string = "Current viewpoint: None. Not in the graph."
    #         return

    #     current_vp_original_id = self.nodeid2viewpoint[self.curr_viewpoint]
    #     neighbors_original = self.graph.get(current_vp_original_id, [])
        
    #     neighbor_ids = [self.viewpoint2nodeid[vp] for vp in neighbors_original]

    #     visited_neighbors = []
    #     unvisited_neighbors = []

    #     for neighbor_id in neighbor_ids:
    #         if neighbor_id in self.visited_nodes:
    #             visited_neighbors.append(neighbor_id)
    #         else:
    #             unvisited_neighbors.append(neighbor_id)

    #     # Sort for deterministic output
    #     visited_neighbors.sort(key=lambda nid: int(nid[5:]))
    #     unvisited_neighbors.sort(key=lambda nid: int(nid[5:]))

    #     status_parts = [f"Current viewpoint: {self.curr_viewpoint}."]
    #     if visited_neighbors:
    #         status_parts.append(f"Visited neighbors: {', '.join(visited_neighbors)}.")
    #     if unvisited_neighbors:
    #         status_parts.append(f"Unvisited neighbors: {', '.join(unvisited_neighbors)}.")
        
    #     self.visit_status_string = " ".join(status_parts)


    def reset(self):
        self.graph = defaultdict(list)

        self.viewpoint2nodeid = {}      # viewpoint_id -> numbered node_id (e.g., 'v0')
        self.nodeid2viewpoint = {}      # numbered node_id -> viewpoint_id
        self.next_node_id = 0           # auto-increment counter for new nodes

        self.curr_viewpoint = None          # current position (numbered node_id)
        self.trajectory = []
        self.connection = ""

        self.visited_nodes = set()          # A set to store visited node IDs
        self.visit_status_string = ""     # The string describing the visit status

        self.all_nodes = set()

        self.curr_summary = ""

    def get_numbered_graph(self):
        """Returns the graph using numbered node IDs."""
        numbered_graph = defaultdict(list)
        for vp, neighbors in self.graph.items():
            vp_id = self.viewpoint2nodeid[vp]
            for neighbor_vp in neighbors:
                neighbor_id = self.viewpoint2nodeid[neighbor_vp]
                numbered_graph[vp_id].append(neighbor_id)
        return numbered_graph

    def bfs_shortest_path(self, start_vp, end_vp):
        if start_vp not in self.graph or end_vp not in self.graph:
            return None

        visited = {start_vp: None}
        queue = deque([start_vp])

        while queue:
            current_node = queue.popleft()

            if current_node == end_vp:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = visited[current_node]
                return path[::-1]

            for neighbor in self.graph[current_node]:
                if neighbor not in visited:
                    visited[neighbor] = current_node
                    queue.append(neighbor)

        return None