import os
import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import logging
from getdown import Hell, SCREEN_WIDTH, SCREEN_HEIGHT, DEADLY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            nn.Linear(self.state_size, 128),  # 第一层
            nn.ReLU(),
            nn.Linear(128, 64),  # 第二层
            nn.ReLU(),
            nn.Linear(64, self.action_size)  # 输出层
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
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        if os.path.exists(file_name):
            self.model.load_state_dict(torch.load(file_name))
            logger.info(f"Model loaded from {file_name}")
        else:
            logger.info(f"No model found at {file_name}, starting from scratch.")


# 环境类的示例
class Env:
    def __init__(self, hell):
        self.hell = hell  # 创建游戏实例
        self.state_size = 18 # 根据状态特征数量调整
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
        reward = self.compute_reward(action)
        done = self.hell.end

        return state, reward, done, {}

    def compute_reward(self, action):
        reward = self.hell.score  # 基于当前分数的奖励
        body = self.hell.body
        barrier = self.hell.barrier

        target_y = body.y + body.h + 2
        matching_barriers = [ba for ba in barrier
                             if ba.rect.y == target_y and ba.rect.x < body.x < (
                                     ba.rect.x + ba.rect.width)]

        # 判断物体所处的位置控制在100~400之间
        if 100 < body.y < 400:
            reward += 6
        elif 150 < body.y < 350:
            reward += 8
        elif 200 < body.y < 300:
            reward += 10
        else:
            reward -= 5

        # 当物体成功离开本级台阶或达到下面某级台阶时，可以提供额外奖励。
        if matching_barriers:
            left_distance = body.x - matching_barriers[0].rect.x
            right_distance = matching_barriers[0].rect.x + matching_barriers[0].rect.width - body.x - body.h
            # 说明在台面上移动
            if left_distance < right_distance and action == 0:
                reward += 4
            elif left_distance > right_distance and action == 1:
                reward += 4
            else:
                reward -= 2

        thres_hold = 100
        matching_barriers = [ba for ba in barrier
                             if 0 < (ba.rect.y - body.y) < thres_hold and ba.rect.x < body.x < (
                                     ba.rect.x + ba.rect.width)]

        # 对于不良行为，给予负奖励。例如，下方快碰到带刺的障碍时
        if matching_barriers and matching_barriers[0].type == DEADLY:
            reward -= 7

        # 当物体到达新的区域或发现新的障碍物时，可以给予奖励。
        if self.preview_barrier_num < len(self.hell.barrier):
            self.preview_barrier_num = len(self.hell.barrier)
            reward += 2
        else:
            reward -= 1

        # 增加下落时朝向障碍物的奖励
        falling_towards_barrier = any(
            ba.rect.x < body.x < (ba.rect.x + ba.rect.width) and ba.rect.y > body.y
            for ba in barrier
        )
        if falling_towards_barrier:
            reward += 3

        # 为每个时间步骤提供小的正奖励，以鼓励持续进行游戏。
        reward += 5
        return reward

    def get_state(self):
        state = []
        state.append(self.hell.body.x)  # 玩家 x 坐标
        state.append(self.hell.body.y)  # 玩家 y 坐标
        state.append(len(self.hell.barrier))  # 障碍物数量

        # 记录最多 max_barriers 个障碍物的信息
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
    env = Env(Hell("是男人就下一百层", (SCREEN_WIDTH, SCREEN_HEIGHT), 60, True))  # 初始化强化学习环境
    agent = DQNAgent(env.state_size, env.action_size)  # 创建 DQN 代理

    model_path = "getdown_hell_model.h5"  # 你可以根据需要更改路径
    agent.load_model(model_path)

    total_steps = 0  # 初始化总步数
    total_game_num = 0
    rewards = []  # 用于记录每个回合的总奖励

    # 记录之前最好的成绩
    best_score = 0

    try:

        state = env.reset()
        while True:  # 无限训练

            state = np.reshape(state, [1, env.state_size])
            total_reward = 0  # 每个回合的总奖励

            #for time in range(1000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10  # 奖励调整
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1
            total_reward += reward  # 更新总奖励
            rewards.append(total_reward)

            # rewards 只保留一万条记录
            if len(rewards) > 10000:
                rewards.pop(0)

            if done:
                # 获取当前回合的得分
                current_score = env.hell.score
                logger.info(
                    f"Total game num: {total_game_num}, Total Steps: {total_steps}, Total score: {current_score}, Epsilon: {agent.epsilon:.7f}")

                # 判断当前得分是否超过阈值
                if current_score >= best_score:
                    logger.info(f'Saving model to getdown_hell_model.h5')
                    agent.save_model("getdown_hell_model.h5")
                    best_score = current_score
                else:
                    logger.info('Model not saved: performance did not meet threshold.')

                total_steps = 0
                total_game_num += 1
                env.hell.reset()

            if len(agent.memory) > 32:
                agent.replay(32)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Saving model...")
        agent.save_model("getdown_hell_model.h5")  # 保存模型

    # 绘制奖励线图
    plt.plot(rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("training_rewards.png", format='png')
