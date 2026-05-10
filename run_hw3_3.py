import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import copy
from collections import deque
import pytorch_lightning as pl
from Gridworld import Gridworld

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return 0

class LitDQN(pl.LightningModule):
    def __init__(self, mem_size=1000, batch_size=200, gamma=0.9, epsilon=0.3, sync_freq=500, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.net = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
        self.target_net = copy.deepcopy(self.net)
        
        self.buffer = deque(maxlen=mem_size)
        self.game = Gridworld(size=4, mode='random')
        self.state = self.get_state()
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.max_moves = 50
        self.moves = 0
        self.global_step_custom = 0

    def get_state(self):
        state_np = self.game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        return torch.from_numpy(state_np).float()

    def setup(self, stage=None):
        # 訓練開始前，先預熱 (Warm-up) 填滿 Buffer
        # 避免一開始 Buffer 資料不夠而無法組成 mini-batch
        while len(self.buffer) < self.hparams.batch_size:
            action_idx = random.randint(0, 3)
            action = self.action_set[action_idx]
            self.game.makeMove(action)
            next_state = self.get_state()
            reward = self.game.reward()
            done = True if reward > 0 else False
            self.buffer.append((self.state, action_idx, reward, next_state, done))
            self.state = next_state
            self.moves += 1
            if reward != -1 or self.moves > self.max_moves:
                self.game = Gridworld(size=4, mode='random')
                self.state = self.get_state()
                self.moves = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # 1. 與環境互動 (Experience Collection)
        qval = self.net(self.state)
        qval_np = qval.detach().numpy()
        
        if random.random() < self.hparams.epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = np.argmax(qval_np)
            
        action = self.action_set[action_idx]
        self.game.makeMove(action)
        next_state = self.get_state()
        reward = self.game.reward()
        done = True if reward > 0 else False
        
        self.buffer.append((self.state, action_idx, reward, next_state, done))
        self.state = next_state
        self.moves += 1
        
        if reward != -1 or self.moves > self.max_moves:
            self.game = Gridworld(size=4, mode='random')
            self.state = self.get_state()
            self.moves = 0

        self.global_step_custom += 1
        if self.global_step_custom % self.hparams.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            
        # 2. 模型優化 (Optimization) 從 Buffer 取出經驗
        minibatch = random.sample(self.buffer, self.hparams.batch_size)
        state_batch = torch.cat([s for (s, a, r, ns, d) in minibatch])
        action_batch = torch.tensor([a for (s, a, r, ns, d) in minibatch])
        reward_batch = torch.tensor([r for (s, a, r, ns, d) in minibatch], dtype=torch.float32)
        next_state_batch = torch.cat([ns for (s, a, r, ns, d) in minibatch])
        done_batch = torch.tensor([d for (s, a, r, ns, d) in minibatch], dtype=torch.float32)
        
        Q1 = self.net(state_batch)
        with torch.no_grad():
            Q2 = self.target_net(next_state_batch)
            
        Y = reward_batch + self.hparams.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        
        # 【Training Tip 3】使用 Huber Loss (Smooth L1) 取代原本的 MSE Loss，對於大誤差更有魯棒性
        loss = nn.SmoothL1Loss()(X, Y.detach())
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        # 【Training Tip 2】學習率排程 (Learning Rate Scheduler)，隨著訓練時間縮小學習步長
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
        return [optimizer], [scheduler]

def test_model(model, mode='random'):
    game = Gridworld(size=4, mode=mode)
    state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state = torch.from_numpy(state_).float()
    status = 1
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    i = 0
    while(status == 1):
        with torch.no_grad():
            qval = model(state)
        action_ = np.argmax(qval.numpy()) 
        action = action_set[action_]
        
        game.makeMove(action)
        state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        
        reward = game.reward()
        if reward != -1:
            status = 2 if reward > 0 else 0
        i += 1
        if i > 15:
            break
            
    return True if status == 2 else False

def run_lightning_dqn():
    print("Initializing PyTorch Lightning DQN...")
    model = LitDQN()
    
    # 定義 Dummy Dataset 來控制總互動 step 數量
    dataset = DummyDataset(50000)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # 【Training Tip 1】啟用梯度裁剪 (gradient_clip_val=1.0) 來預防梯度爆炸
    trainer = pl.Trainer(
        max_steps=50000, 
        gradient_clip_val=1.0,  
        enable_checkpointing=False,
        logger=False
    )
    
    # 進行訓練
    trainer.fit(model, dataloader)
    
    # 測試效能
    print("Testing the model on 1000 random games...")
    wins = sum([test_model(model, mode='random') for _ in range(1000)])
    print(f"PyTorch Lightning DQN Win rate (Random Mode): {wins/1000.0 * 100:.2f}%")

if __name__ == '__main__':
    run_lightning_dqn()
