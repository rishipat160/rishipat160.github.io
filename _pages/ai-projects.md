---
title: "AI Projects"
permalink: /ai-projects/
layout: splash
author_profile: false
classes: wide
---

<div class="projects-container">

<div class="project-card" id="castleescape-rl">
  <h2>CastleEscape-RL: Reinforcement Learning for Grid-Based Escape Game</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-python"></i> PyTorch, Numpy, OpenAI Gym </span>
    <a href="https://github.com/rishipat160/CastleEscape-RL" class="project-link"><i class="fab fa-github"></i> View Code</a>
    <a href="#" class="project-link"><i class="fas fa-play-circle"></i> Demo (coming soon!)</a>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>In this project, I tackled a challenging reinforcement learning problem: teaching an agent to navigate through a castle, avoid or defeat guards, and find the exit. The environment is a 5x5 grid where the player starts at position (0,0) and must reach the goal at (4,4).</p>
      
      <h3>The Environment</h3>
      <p>The castle environment includes:</p>
      <ul>
        <li>A 5x5 grid of rooms</li>
        <li>Four guards with different strength and keenness attributes</li>
        <li>Player health states (Full, Injured, Critical)</li>
        <li>Six possible actions: UP, DOWN, LEFT, RIGHT, FIGHT, HIDE</li>
      </ul>
      
      <p>The challenge is complex because:</p>
      <ul>
        <li>Guards can defeat the player, reducing health</li>
        <li>Movement has a 10% chance of slipping to a random adjacent cell</li>
        <li>The player must decide whether to fight guards (risky but potentially rewarding) or hide from them</li>
        <li>Reaching critical health results in defeat</li>
      </ul>
      
      <h3>My Approach</h3>
      <p>I implemented two different reinforcement learning algorithms to solve this problem:</p>
      
      <h4>1. Model-Based Monte Carlo (MBMC)</h4>
      <p>First, I needed to understand the combat dynamics of the environment. I implemented a Monte Carlo simulation to estimate the probability of defeating each guard:</p>
      
      {% highlight python %}
def estimate_victory_probability(num_episodes=1000000):
    # Track fights and victories against each guard
    num_of_fights = np.zeros(len(env.guards))
    num_of_success = np.zeros(len(env.guards))
    
    for _ in range(num_episodes):
        obs, reward, done, info = env.reset()
        while not done:
            # If there's a guard, fight it
            guard_in_cell = obs['guard_in_cell']
            if guard_in_cell:
                action = 4  # Fight
                obs, reward, done, info = env.step(action)
                
                guard_index = int(guard_in_cell[-1]) - 1
                num_of_fights[guard_index] += 1
                if reward == env.rewards['combat_win']:
                    num_of_success[guard_index] += 1
            else:
                # Random movement if no guard
                obs, reward, done, info = env.step(np.random.randint(4))
    
    # Calculate probabilities
    P = np.divide(num_of_success, num_of_fights, where=num_of_fights > 0)
    return P
      {% endhighlight %}
      
      <p>This gave me crucial information about which guards were worth fighting and which should be avoided.</p>
      
      <h4>2. Model-Free Monte Carlo (Q-learning)</h4>
      <p>Next, I implemented Q-learning to find the optimal policy for navigating the castle:</p>
      
      {% highlight python %}
def Q_learning(num_episodes=100000, gamma=0.9, epsilon=1, decay_rate=0.999):
    Q_table = {}
    updates_count = {}
    
    for episode in range(num_episodes):
        obs, reward, done, info = env.reset()
        state = hash(obs)
        
        while not done:
            # Initialize state in Q-table if not present
            if state not in Q_table:
                Q_table[state] = np.zeros(6)
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(6)
            else:
                action = np.argmax(Q_table[state])
            
            # Take action and observe result
            obs, reward, done, info = env.step(action)
            next_state = hash(obs)
            
            # Initialize next state in Q-table if not present
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(6)
            
            # Update Q-values using learning rate that decreases with experience
            updates_count[(state, action)] = updates_count.get((state, action), 0) + 1
            eta_sa = 1 / (1 + updates_count[(state, action)])
            max_next_Q = np.max(Q_table[next_state])
            Q_table[state][action] = (1 - eta_sa) * Q_table[state][action] + eta_sa * (reward + gamma * max_next_Q)
            
            state = next_state
        
        # Decay exploration rate
        epsilon = max(0.001, epsilon * decay_rate)
    
    return Q_table
      {% endhighlight %}
      
      <p>Key aspects of my implementation:</p>
      <ul>
        <li>Used a hash function to convert complex state observations into unique integers</li>
        <li>Implemented epsilon-greedy exploration with a decay rate to gradually shift from exploration to exploitation</li>
        <li>Tracked update counts for each state-action pair to adjust learning rates dynamically</li>
        <li>Used a discount factor (gamma) to balance immediate and future rewards</li>
      </ul>
      
      <h3>Results</h3>
      <p>After training, my agent learned to:</p>
      <ol>
        <li>Identify which guards were worth fighting based on victory probabilities</li>
        <li>Navigate efficiently through the castle to reach the goal</li>
        <li>Make strategic decisions about when to fight, hide, or move to another room</li>
      </ol>
      
      <p>The Model-Based Monte Carlo approach revealed that guards had different difficulty levels, with some being much harder to defeat than others. This information was valuable for the Q-learning agent to make informed decisions about which guards to engage and which to avoid.</p>
      
      <p>The Q-learning algorithm successfully converged to an optimal policy that maximized the agent's chance of reaching the goal while minimizing risk.</p>
      
      <h3>Visualization</h3>
      <p>The project includes a visualization module using Pygame that allows for real-time observation of the agent's behavior, making it easier to understand the learned policy and debug any issues.</p>
      
    </div>
  </details>
</div>

<div class="project-card" id="hill-climbing-puzzle">
  <h2>GridColorSolver: Hill Climbing Search for Constraint Satisfaction Puzzles</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-python"></i> Python, NumPy, Matplotlib</span>
    <a href="https://github.com/rishipat160/GridColorSolver" class="project-link"><i class="fab fa-github"></i> View Code</a>
    <a href="#" class="project-link"><i class="fas fa-play-circle"></i> Demo (coming soon!)</a>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>In this project, I implemented a hill climbing search algorithm to solve a challenging grid coloring puzzle. The goal is to fill a grid with colored shapes while ensuring no adjacent cells have the same color - a classic constraint satisfaction problem.</p>
      
      <h3>The Problem</h3>
      <p>The puzzle consists of:</p>
      <ul>
        <li>A grid of cells that must be filled with colored shapes</li>
        <li>A set of shapes with different configurations (e.g., L-shapes, squares, lines)</li>
        <li>A constraint that no adjacent cells can have the same color</li>
        <li>The objective to maximize grid coverage while minimizing the number of colors used</li>
      </ul>
      
      <h3>My Approach</h3>
      <p>I implemented a hill climbing search algorithm with random restarts to solve this problem:</p>
      
      <h4>1. Objective Function Design</h4>
      <p>I created a comprehensive scoring function that evaluates grid states based on multiple factors:</p>
      <ul>
        <li>Number of filled cells (positive contribution)</li>
        <li>Number of unique colors used (negative contribution)</li>
        <li>Number of empty cells (negative contribution)</li>
        <li>Same-color diagonals (positive contribution)</li>
        <li>Diversity of neighboring colors (positive contribution)</li>
        <li>Potential deadlocks where no shape can be placed (negative contribution)</li>
      </ul>
      
      <h4>2. Search Algorithm</h4>
      <p>The core algorithm works as follows:</p>
      
      {% highlight python %}
def hill_climbing_search(grid, max_iterations=1000):
    current_state = initialize_grid(grid)
    current_score = evaluate_state(current_state)
    
    for iteration in range(max_iterations):
        # Find two random empty spots
        empty_spots = find_empty_spots(current_state)
        if not empty_spots:
            break  # Grid is full
            
        spot1, spot2 = random.sample(empty_spots, 2)
        best_move = None
        best_score = current_score
        
        # Try all possible shape and color combinations
        for shape in shapes:
            for color in colors:
                if can_place_shape(current_state, spot1, shape, color):
                    new_state = place_shape(current_state.copy(), spot1, shape, color)
                    new_score = evaluate_state(new_state)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_move = (spot1, shape, color)
        
        # If no valid move found, try random restart
        if not best_move:
            current_state = random_restart(current_state)
            current_score = evaluate_state(current_state)
        else:
            # Execute the best move
            current_state = place_shape(current_state, *best_move)
            current_score = best_score
            
    return current_state
      {% endhighlight %}
      
      <h4>3. Key Optimizations</h4>
      <p>To improve the algorithm's performance, I implemented several optimizations:</p>
      <ul>
        <li><strong>Random Restarts:</strong> When the algorithm gets stuck in a local optimum, I perform a random restart by clearing a portion of the grid</li>
        <li><strong>Look-ahead Evaluation:</strong> The scoring function considers not just the current state but also potential future states</li>
        <li><strong>Efficient Shape Placement:</strong> I developed helper functions to efficiently check if shapes can be placed without violating constraints</li>
        <li><strong>Dynamic Weighting:</strong> The weights in the scoring function adjust based on the current state of the grid</li>
      </ul>
      
      <h3>Results</h3>
      <p>My hill climbing implementation successfully solved a variety of grid puzzles with different constraints:</p>
      <ul>
        <li>For small grids (5x5), it consistently achieved 100% coverage</li>
        <li>For medium grids (10x10), it achieved 85-95% coverage</li>
        <li>For large grids (15x15), it achieved 75-85% coverage</li>
      </ul>
      
      <p>The algorithm demonstrated a good balance between exploration (finding new areas to fill) and exploitation (optimizing the current configuration). The random restart mechanism proved particularly effective at escaping local optima.</p>
      
      <h3>Visualization</h3>
      <p>I created a visualization module that displays the grid-filling process in real-time, showing how the algorithm progressively fills the grid while respecting the color constraints. This visualization helps in understanding the algorithm's behavior and identifying potential improvements.</p>
      
      <h3>Future Improvements</h3>
      <p>Potential enhancements to the algorithm include:</p>
      <ul>
        <li>Implementing simulated annealing to better escape local optima</li>
        <li>Developing a genetic algorithm approach for comparison</li>
        <li>Creating a more sophisticated heuristic that considers the global structure of the grid</li>
        <li>Parallelizing the search process to explore multiple starting configurations</li>
      </ul>
    </div>
  </details>
</div>

<div class="project-card" id="fashion-mnist-classifier">
  <h2>Fashion-MNIST Classifier: Deep Learning for Image Classification</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-python"></i> PyTorch, NumPy, Matplotlib, OpenCV</span>
    <a href="https://github.com/rishipat160/Fashion-MNIST-Classifier" class="project-link"><i class="fab fa-github"></i> View Code</a>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>In this project, I developed and compared two neural network architectures for classifying clothing items from the Fashion-MNIST dataset. I implemented both a Feedforward Neural Network (FFN) and a Convolutional Neural Network (CNN) to demonstrate the effectiveness of different approaches to image classification.</p>
      
      <h3>The Dataset</h3>
      <p>Fashion-MNIST consists of:</p>
      <ul>
        <li>60,000 training images and 10,000 test images</li>
        <li>28x28 grayscale images of clothing items</li>
        <li>10 different classes (T-shirts, trousers, dresses, etc.)</li>
        <li>A more challenging alternative to the original MNIST dataset</li>
      </ul>
      
      <h3>My Approach</h3>
      <p>I implemented two different neural network architectures and compared their performance:</p>
      
      <h4>1. Feedforward Neural Network (FFN)</h4>
      <p>I designed a multi-layer FFN with the following architecture:</p>
      
      {% highlight python %}
class FF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
      {% endhighlight %}
      
      <p>Key features of the FFN:</p>
      <ul>
        <li>5 fully connected layers with decreasing sizes</li>
        <li>Batch normalization after each layer except the output</li>
        <li>ReLU activation functions</li>
        <li>Dropout for regularization</li>
      </ul>
      
      <h4>2. Convolutional Neural Network (CNN)</h4>
      <p>I implemented a CNN with the following architecture:</p>
      
      {% highlight python %}
class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.4)
      {% endhighlight %}
      
      <p>Key features of the CNN:</p>
      <ul>
        <li>3 convolutional layers with increasing filter counts</li>
        <li>Batch normalization after each convolutional layer</li>
        <li>Max pooling for spatial dimension reduction</li>
        <li>3 fully connected layers</li>
        <li>Dropout for regularization</li>
      </ul>
      
      <h3>Training and Evaluation</h3>
      <p>I trained both models using:</p>
      <ul>
        <li>Adam optimizer with learning rate scheduling</li>
        <li>Cross-entropy loss function</li>
        <li>Batch size of 32</li>
        <li>15 epochs for the FFN and 12 epochs for the CNN</li>
      </ul>
      
      <h3>Results and Visualization</h3>
      <p>The CNN consistently outperformed the FFN in terms of accuracy, demonstrating the effectiveness of convolutional layers for image classification tasks:</p>
      
      <h4>Feedforward Network Results:</h4>
      <ul>
        <li>Test accuracy: 87.7%</li>
        <li>Training subset accuracy: 90.7%</li>
        <li>Mean accuracy: 88.2%</li>
      </ul>
      
      <h4>Convolutional Network Results:</h4>
      <ul>
        <li>Test accuracy: 90.18%</li>
        <li>Training subset accuracy: 94.4%</li>
        <li>Mean accuracy: 90.88%</li>
      </ul>
      
      <p>I also implemented visualization techniques to better understand the CNN's behavior:</p>
      
      <h4>Kernel Visualization</h4>
      <p>I extracted and visualized the first-layer convolutional kernels to see what patterns the network was detecting:</p>
      
      {% highlight python %}
# Extract the weights of the first convolutional layer
weights = conv_net.conv1.weight.data
num_kernels = weights.shape[0]

# Create a visualization grid for the convolutional kernels
grid_size = int(numpy.ceil(numpy.sqrt(num_kernels)))
fig = plt.figure(figsize=(10, 10))

for i in range(num_kernels):
    kernel = weights[i, 0].detach().numpy()
    # Normalize kernel values to [0,1] for visualization
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(kernel, cmap='gray')
    plt.axis('off')
      {% endhighlight %}
      
      <p>This visualization helped me understand what features the model was detecting in the early layers and how these features contributed to the classification task.</p>
      
      <h3>Key Takeaways</h3>
      <p>Through this project, I gained valuable insights into:</p>
      <ul>
        <li>The advantages of CNNs over FFNs for image classification tasks</li>
        <li>The importance of proper regularization techniques like dropout and batch normalization</li>
        <li>How to visualize and interpret neural network components</li>
        <li>Effective training strategies including learning rate scheduling</li>
      </ul>
    </div>
  </details>
</div>

</div>


