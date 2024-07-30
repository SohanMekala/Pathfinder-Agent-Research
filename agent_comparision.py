import random

#important values
node_val_lower_bound = 10
node_val_upper_bound = 20
edge_val_lower_bound = 1
edge_val_upper_bound = 5

num_nodes = 20
max_duration = 400
request_count = 80

num_episodes = 250

#request class
class Request:
    def __init__(self, start_node, end_node, cost):
        self.start_node = start_node
        self.end_node = end_node
        self.cost = cost

#generate graph
def generate_large_graph(num_nodes):
    nodes = [chr(ord('A') + i) for i in range(num_nodes)]
    values = {node: random.randint(node_val_lower_bound, node_val_upper_bound) for node in nodes}
    edges = {}

    #ensure that the graph is connected
    for i in range(num_nodes - 1):
        weight = random.randint(edge_val_lower_bound, edge_val_upper_bound)
        edges[(nodes[i], nodes[i + 1])] = weight

    #add additional random edges
    additional_edges = int(num_nodes*2)
    while len(edges) < additional_edges:
        start, end = random.sample(nodes, 2)
        if (start, end) not in edges and (end, start) not in edges:
            weight = random.randint(edge_val_lower_bound, edge_val_upper_bound)
            edges[(start, end)] = weight

    return nodes, values, edges

def edgeBasedHeuristic(req):
    #djikstras_algorithm
    start = req.start_node
    end = req.end_node
    cost = req.cost

    #final metrics
    global successful_requests
    global elapsed_time

    #necessary components for djikstras algorithm
    unvisited = []
    shortestDist = {}
    prevNode = {}

    #overall initialization
    for node in nodes:
        unvisited.append(node)

        if node == start:
            shortestDist[node] = 0
        else:
            shortestDist[node] = float('inf')
        
        prevNode[node] = None

    workingNode = start

    #actual algorithm
    while workingNode != end:
        neighbors = {}

        for edge in edges:
            if workingNode in edge:

                neighbor = edge[1-edge.index(workingNode)]
                if neighbor in unvisited:
                    
                    if shortestDist[workingNode]+edges[edge] < shortestDist[neighbor]:
                        shortestDist[neighbor] = shortestDist[workingNode]+edges[edge]
                        prevNode[neighbor] = workingNode

        unvisited.remove(workingNode)

        #greedy aproach to find closest node
        closestNeighborDist = float('inf')
        closestNeighbor = None

        for neighbor in unvisited:
            if shortestDist[neighbor] < closestNeighborDist:
                closestNeighbor = neighbor
                closestNeighborDist = shortestDist[neighbor]
        
        workingNode = closestNeighbor

    #backtracking
    path = []
    node = end
    path_sum = 0
    while node is not None:
        path_sum+=values[node]
        path.append(node)
        node = prevNode[node]
    path.reverse()

    duration = shortestDist[end]

    #updating elapsed time, computing resources for nodes, and successful request count
    if((max_duration-elapsed_time)<duration):
        return False
    elif(path_sum<cost):
        return False
    else:
        i = 0
        while cost!=0:
            if i==len(path):
                i=0
            if(values[path[i]]>=1):
                values[path[i]]-=1
                cost-=1
            i+=1
        successful_requests+=1
        elapsed_time+=duration
        return True

#tabular q-learning approach
def initialize_q_table(nodes):
    return {(start, end): {action: 0 for action in nodes} for start in nodes for end in nodes}

#choose action based on epsilon-greedy policy
def choose_action(state, q_table, epsilon):
    #epsilon-greedy policy to balance exploration and exploitation
    if random.uniform(0, 1) < epsilon:
        #exploration
        return random.choice(list(q_table[state].keys()))
    else:
        #exploitation
        return max(q_table[state], key=q_table[state].get)

#constantly update q-table
def update_q_table(q_table, state, action, next_state, reward, alpha, gamma):
    current_q = q_table[state][action]
    max_next_q = max(q_table[next_state].values())
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    q_table[state][action] = new_q

#actual path agent
def reinforcement_learning(req, q_table, epsilon, alpha, gamma, num_episodes):
    start = req.start_node
    end = req.end_node
    cost = req.cost

    global successful_requests
    global elapsed_time

    best_success = False
    best_duration = float('inf')
    best_path_sum = float('inf')

    for _ in range(num_episodes):
        #reset state for each episode
        current_state = start
        path = [current_state]
        path_sum = values[current_state]
        total_duration = 0
        path_found = False

        #perform reinforcement learning process until end node is reached or episode limit is hit
        while current_state != end:
            action = choose_action((current_state, end), q_table, epsilon)
            if (current_state, action) in edges:
                duration = edges[(current_state, action)]
            elif (action, current_state) in edges:
                duration = edges[(action, current_state)]
            else:
                #if there's no direct edge, penalize this action
                update_q_table(q_table, (current_state, end), action, (action, end), -10, alpha, gamma)
                break  #exit this episode early

            total_duration += duration
            path_sum += values[action]
            path.append(action)

            #check if total duration exceeds maximum allowed duration
            if total_duration > max_duration - elapsed_time:
                update_q_table(q_table, (current_state, end), action, (action, end), -5, alpha, gamma)
                break  #exit this episode early

            #compute the reward
            reward = 1 if path_sum >= cost else -1
            update_q_table(q_table, (current_state, end), action, (action, end), reward, alpha, gamma)

            current_state = action

        #final decision and following updates
        if path_sum >= cost and total_duration < best_duration:
            best_success = True
            best_duration = total_duration
            best_path_sum = path_sum

    # apply the best result found
    if best_success:
        for node in path:
            if cost > 0:
                deduction = min(values[node], cost)
                values[node] -= deduction
                cost -= deduction
        successful_requests += 1
        elapsed_time += best_duration
    return best_success

#main execution
nodes, values, edges = generate_large_graph(num_nodes)

#generate requests
requests = []
for _ in range(request_count):
    start_node, end_node = random.sample(nodes, 2)
    cost = random.randint(5, 10) 
    request = Request(start_node, end_node, cost)
    requests.append(request)

#heuristic approach
successful_requests = 0
elapsed_time = 0
heuristic_results = []

#pass requests
for request in requests:
    result = edgeBasedHeuristic(request)
    heuristic_results.append(result)

heuristic_success_rate= successful_requests / request_count

#reset for rl, only the values are being reset
values, _, _ = generate_large_graph(num_nodes)
successful_requests = 0
elapsed_time = 0

#reinforcement Learning parameters
epsilon = 0.1
alpha = 0.1
gamma = 0.9

q_table = initialize_q_table(nodes)
rl_results = []

#pass requests
for request in requests:
    result = reinforcement_learning(request, q_table, epsilon, alpha, gamma, num_episodes)
    rl_results.append(result)

rl_success_rate = successful_requests / request_count

#final metrics
print(num_episodes)
print(f"Heuristic Approach Success Rate: {heuristic_success_rate}%")
print(f"Reinforcement Learning Success Rate: {rl_success_rate}%")
