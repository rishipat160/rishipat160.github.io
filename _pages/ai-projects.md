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
    <span class="project-tech"><i class="fab fa-python"></i> PyTorch, OpenAI Gym, </span>
    <a href="https://github.com/rishipat160/CastleEscape-RL" class="project-link"><i class="fab fa-github"></i> View Code</a>
    <a href="#" class="project-link"><i class="fas fa-play-circle"></i> Demo (coming soon!)</a>
  </div>

  <details open>
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

<div class="project-card" id="project-2">
  <h2>Project 2: NLP for Medical Records</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-python"></i> Coming soon</span>
    <a href="#" class="project-link"><i class="fab fa-github"></i> View Code</a>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>Coming soon...</p>
    </div>
  </details>
</div>

</div>


