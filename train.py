from environment.cnn_environment import CNNTuningEnvironment
from agent.dqn_agent import DQNAgent
import numpy as np
from utils.hyperparameters import DROPOUT_RATES, CONV_KERNEL_SIZES, EPSILON_START
import matplotlib.pyplot as plt
import time as tm

accuracy_per_episode = []  # 存储每个episode的最佳准确率
time_per_episode = []  # 存储每个episode的时间（可以用秒或分钟等单位）
start_time = tm.time()

def print_and_save(text, file_path):
    print(text)
    with open(file_path, "a") as file:
        file.write(text + "\n")

file_path = "dqn_cnn_tuning_log.txt"
def train_dqn(episode_count):
    n_actions = len(CONV_KERNEL_SIZES) * len(DROPOUT_RATES) * len(np.linspace(0.0001, 0.01, num=10)) * len([128, 256, 512, 1024]) * len([0.3, 0.4, 0.5, 0.6, 0.7]) * len([16, 32, 64, 128]) * len(np.linspace(0.5, 1, num=5))
    state_size = 6  # 假设包括kernel size index, fc_layer_size index, dropout rate index, batch size index, learning rate index, and momentum index
    agent = DQNAgent(state_size, n_actions)
    env = CNNTuningEnvironment()

    # batch_size = 32

    for e in range(episode_count):
        env.iteration_count = 0  # 重置迭代计数
        state = env.reset()  # 确保这里返回的是正确形状的状态
        for time in range(300):  # 每个episode的时间步数
            action = agent.act(state)  # 选择动作
            next_state, reward, done = env.step(action)  # 执行动作并获取下一个状态和奖励
            reward = reward if not done else -10  # 如果动作结束了episode，则给予负奖励
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # 执行动作后，打印当前准确率和最佳准确率以及最佳超参数
            print_and_save(f"当前准确率: {env.last_performance}", file_path)
            print_and_save(f"当前模型状态: {env.last_hyperparams}", file_path)
            # print(env.last_hyperparams)
            print_and_save(f"最佳准确率: {env.best_accuracy}", file_path)
            print_and_save(f"最佳模型状态: {env.best_hyperparams}\n", file_path)
            # print(env.best_hyperparams)

            accuracy_per_episode.append(env.best_accuracy)  # 将最佳准确率添加到列表中

            # 计算当前episode结束时的累积时间（以秒为单位）
            elapsed_time = tm.time() - start_time
            time_per_episode.append(elapsed_time)

            if done:
                print_and_save("_____________________________________________________________", file_path)
                print_and_save("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, episode_count, time, agent.epsilon), file_path)
                # print("_____________________________________________________________")
                # print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, episode_count, time, agent.epsilon))
                break
            if len(agent.memory) > agent.batch_size:
                print_and_save("调用replay方法", file_path)
                # print("调用replay方法")
                agent.replay()
        if e % 10 == 0:
            agent.save("./save/dqn_cnn_tuning_{}.pt".format(e))

if __name__ == "__main__":
    episode_count = 100
    train_dqn(episode_count)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy_per_episode, label='Validation Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Episode')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(time_per_episode, accuracy_per_episode, label='Accuracy over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig('validation_accuracy.png')
    plt.show()

