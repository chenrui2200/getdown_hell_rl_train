import os

import numpy as np
import random

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from matplotlib import pyplot as plt

from getdown import Hell, SCREEN_WIDTH, SCREEN_HEIGHT


class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),  # 第一层
            nn.ReLU(),
            nn.Linear(24, 24),  # 第二层
            nn.ReLU(),
            nn.Linear(24, self.action_size)  # 输出层
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机选择动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            act_values = self.model(state_tensor)
        return np.argmax(act_values.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f = target_f.squeeze()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.loss_fn(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)  # 保存模型的参数

    def load_model(self, file_name):
        if os.path.exists(file_name):
            self.model.load_state_dict(torch.load(file_name))  # 加载模型的参数
            print(f"Model loaded from {file_name}")
        else:
            print(f"No model found at {file_name}, starting from scratch.")



# 环境类的示例
class Env:
    def __init__(self, hell):
        self.hell = hell # 创建游戏实例
        self.state_size = 18  # 根据状态特征数量调整
        self.action_size = 3  # 左, 右, 不动
        self.preview_barrier_num = 0;

    def reset(self):
        # 重置游戏
        self.hell.reset()
        return self.get_state()


    def step(self, action):
        if action == 0:  # Move left
            self.hell.move(pygame.K_LEFT)
        elif action == 1:  # Move right
            self.hell.move(pygame.K_RIGHT)
        else:  # Stay still
            self.hell.unmove(None)

        self.hell.update(pygame.time.get_ticks())  # 更新游戏状态

        # 获取状态、奖励和结束信息
        state = self.get_state()
        reward = self.compute_reward()
        done = self.hell.end

        return state, reward, done, {}

    def compute_reward(self):
        reward = self.hell.score  # 基于当前分数的奖励

        # 当物体成功离开本级台阶或达到下面某级台阶时，可以提供额外奖励。
        if self.agent_leave_barrier():
            reward += 10  # 避免障碍物

        # 对于不良行为，给予负奖励。例如，当物体碰到带刺障碍失败时
        if self.agent_hit_badbarrier():
            reward -= 10

        # 当物体到达新的区域或发现新的障碍物时，可以给予奖励。
        if self.preview_barrier_num < len(self.hell.barrier):
            self.preview_barrier_num = self.hell.barrier
            reward += 3
        else:
            reward -= 1

        # 为每个时间步骤提供小的正奖励，以鼓励持续进行游戏。
        reward += 0.1
        return reward

    def get_state(self):
        state = []
        state.append(self.hell.body.x)  # 玩家 x 坐标
        state.append(self.hell.body.y)  # 玩家 y 坐标
        state.append(len(self.hell.barrier))  # 障碍物数量

        # 记录最多 2 个障碍物的信息
        max_barriers = 5
        for i in range(max_barriers):
            if i < len(self.hell.barrier):
                ba = self.hell.barrier[i]
                state.append(ba.rect.x)  # 障碍物 x 坐标
                state.append(ba.rect.y)  # 障碍物 y 坐标
                state.append(ba.type)  # 障碍物类型
            else:
                # 如果没有障碍物，用零填充
                state.extend([0, 0, 0])

        # 确保状态的长度与 state_size 一致
        return np.array(state)


# 训练主循环
if __name__ == "__main__":
    env = Env(Hell("是男人就下一百层", (SCREEN_WIDTH, SCREEN_HEIGHT)))  # 初始化强化学习环境
    agent = DQNAgent(env.state_size, env.action_size)  # 创建 DQN 代理

    model_path = "getdown_hell_model.h5"  # 你可以根据需要更改路径
    agent.load_model(model_path)

    total_steps = 0  # 初始化总步数
    rewards = []  # 用于记录每个回合的总奖励

    try:
        while True:  # 无限训练
            state = env.reset()
            state = np.reshape(state, [1, env.state_size])
            total_reward = 0  # 每个回合的总奖励

            for time in range(500):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10  # 奖励调整
                next_state = np.reshape(next_state, [1, env.state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_steps += 1
                total_reward += reward  # 更新总奖励

                if done:
                    rewards.append(total_reward)  # 记录每个回合的总奖励
                    print(f"Score: {time}, Total Steps: {total_steps}, Epsilon: {agent.epsilon:.2}")
                    break

                # 在每1000步时保存模型
                if total_steps % 1000 == 0:
                    agent.save_model("getdown_hell_model.h5")  # 保存模型

            if len(agent.memory) > 32:
                agent.replay(32)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save_model("getdown_hell_model.h5")  # 保存模型


    # 绘制奖励线图
    plt.plot(rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
