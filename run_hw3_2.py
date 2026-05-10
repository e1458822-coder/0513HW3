import numpy as np
import torch
import random
import copy
from matplotlib import pylab as plt
from Gridworld import Gridworld
from collections import deque

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

def test_model(model, mode='player', display=False):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state = torch.from_numpy(state_).float()
    status = 1
    
    while(status == 1):
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_) 
        action = action_set[action_]
        
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
            else:
                status = 0
        i += 1
        if (i > 15):
            break
    
    win = True if status == 2 else False
    return win

def run_double_dqn():
    print("Running Double DQN on player mode...")
    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4)
    )
    
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3
    epochs = 3000
    losses = []
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    sync_freq = 500
    j = 0
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        while(status == 1): 
            j += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            exp =  (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                
                Q1 = model(state1_batch) 
                
                # Double DQN target evaluation
                with torch.no_grad():
                    Q1_next = model(state2_batch)
                    action_next = torch.argmax(Q1_next, dim=1)
                    Q2_next = model2(state2_batch)
                    Q2_next_action_values = Q2_next.gather(dim=1, index=action_next.unsqueeze(dim=1)).squeeze()
                
                Y = reward_batch + gamma * ((1-done_batch) * Q2_next_action_values)
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
                    
            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0

    wins = sum([test_model(model, mode='player') for _ in range(1000)])
    print(f"Double DQN Win rate: {wins/1000.0 * 100:.2f}%")
    
    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Double DQN Loss (Player Mode)")
    plt.savefig('double_dqn_loss.png')


class DuelingDQN(torch.nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(64, 150)
        self.fc2 = torch.nn.Linear(150, 100)
        self.value_stream = torch.nn.Linear(100, 1)
        self.advantage_stream = torch.nn.Linear(100, 4)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

def run_dueling_dqn():
    print("Running Dueling DQN on player mode...")
    model = DuelingDQN()
    
    # Also using a target network for stability
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3
    epochs = 3000
    losses = []
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    sync_freq = 500
    j = 0
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        while(status == 1): 
            j += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            exp =  (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                
                Q1 = model(state1_batch) 
                with torch.no_grad():
                    Q2_next = model2(state2_batch)
                
                # Standard Target evaluation
                Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2_next,dim=1)[0])
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
                    
            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0

    wins = sum([test_model(model, mode='player') for _ in range(1000)])
    print(f"Dueling DQN Win rate: {wins/1000.0 * 100:.2f}%")
    
    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Dueling DQN Loss (Player Mode)")
    plt.savefig('dueling_dqn_loss.png')

if __name__ == '__main__':
    run_double_dqn()
    run_dueling_dqn()
