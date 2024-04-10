from environment.cnn_environment import CNNTuningEnvironment
from agent.dqn_agent import DQNAgent
import numpy as np
from utils.hyperparameters import DROPOUT_RATES, CONV_KERNEL_SIZES, FC_NODES, BATCH_SIZES, LEARNING_RATES, MOMENTUMS
import matplotlib.pyplot as plt
import time as tm

accuracy_per_episode = []  # 存储每个episode的最佳准确率
losses = []  # 在训练循环开始前初始化损失记录列表

def print_and_save(text, file_path):
    print(text)
    with open(file_path, "a") as file:
        file.write(text + "\n")

file_path = "dqn_cifar1.txt"
def train_dqn(episode_count):
    n_actions = len(CONV_KERNEL_SIZES) * len(DROPOUT_RATES) * len(LEARNING_RATES) * len(FC_NODES) * len(BATCH_SIZES) * len(MOMENTUMS)
    # n_actions = len(CONV_KERNEL_SIZES) * len(DROPOUT_RATES) * len(np.linspace(0.0001, 0.01, num=10)) * len([128, 256, 512, 1024]) * len([16, 32, 64, 128]) * len(np.linspace(0.5, 1, num=5))
    state_size = 6  # 假设包括kernel size index, fc_layer_size index, dropout rate index, batch size index, learning rate index, and momentum index
    agent = DQNAgent(state_size, n_actions)
    start_episode, start_accuracy, start_hyperparams, epsilon = agent.load("save/dqn_cnn_tuning_10.pt")  # 替换成你之前保存的模型路径
    # start_episode, start_accuracy, start_hyperparams, epsilon = 0, 0, {}, 1.0
    env = CNNTuningEnvironment()
    env.best_accuracy = start_accuracy
    env.best_hyperparams = start_hyperparams
    env.epsilon = epsilon


    # batch_size = 32
    # 一个episode通常指的是从环境（env）的初始状态开始，到达终止状态的一系列动作和状态的序列。
    # print("开始训练")
    for e in range(start_episode, episode_count):
        agent.episode_count = e
        # print(f"当前episode_count:{agent.episode_count}")
        env.iteration_count = 0  # 重置迭代计数
        state = env.reset()  # 确保这里返回的是正确形状的状态
        print(f"last_performance:{env.last_performance}")

        for iteration in range(max_iterations):  # 每个episode的时间步数
            action = agent.act(state)  # 选择动作
            # print("循环中的action:", action)
            next_state, reward, done = env.step(action)  # 执行动作并获取下一个状态和奖励
            # print("循环中的next_state:", next_state)
            # 如果done是False，则reward保持原值不变。如果done是True，则reward被设置为-10。
            # 这样做的目的是为了让智能体尽可能快地学会如何结束游戏/过程。
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)  # 记住状态、动作、奖励、下一个状态和是否结束
            state = next_state

            print_and_save(f"len(agent.memory):{len(agent.memory)}", file_path)
            print_and_save(f"agent.batch_size:{agent.batch_size}", file_path)
            # 执行动作后，打印当前准确率和最佳准确率以及最佳超参数
            print_and_save(f"当前准确率: {env.last_performance}", file_path)
            print_and_save(f"当前模型状态: {env.last_hyperparams}", file_path)
            # print(env.last_hyperparams)
            print_and_save(f"最佳准确率: {env.best_accuracy}", file_path)
            print_and_save(f"最佳模型状态: {env.best_hyperparams}\n", file_path)
            # print(env.best_hyperparams)
            agent.best_accuracy = env.best_accuracy
            agent.best_hyperparams = env.best_hyperparams

            accuracy_per_episode.append(env.best_accuracy)  # 将最佳准确率添加到列表中

            if done:
                print_and_save("_____________________________________________________________", file_path)
                print_and_save("episode: {}/{}, 在第{}次迭代时终止, epsilon: {:.2}".format(e, episode_count, iteration + 1, agent.epsilon), file_path)
                # print("_____________________________________________________________")
                # print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, episode_count, time, agent.epsilon))
                break

            # print(f"agent.model.state_dict()的第一个元素：{agent.model.state_dict()['fc1.weight'][0]}")
            # 学习：定期从经验回放池中抽取一批经验，并使用 DQNAgent 类的 replay 方法来更新网络参数。
            if len(agent.memory) > agent.batch_size:
                print_and_save("调用replay方法", file_path)
                # print("调用replay方法")
                loss = agent.replay()
                losses.append(loss)  # 记录每个epoch的损失值
                accuracy_per_episode.append(env.last_performance)  # 将最佳准确率添加到列表中
        if e % 2 == 0:
            agent.save("./save/dqn_cnn_tuning_{}.pt".format(e))

if __name__ == "__main__":
    max_iterations = 10  # 最大迭代次
    episode_count = 100  # episode的数量
    train_dqn(episode_count)

    plt.plot(accuracy_per_episode, marker='o', linestyle='-', color='b', label='Accuracy')  # 修改label为英文
    plt.title('DQN CNN Hyperparameter Tuning')  # 将标题改为英文
    plt.xlabel('Episode')  # X轴标签已经是英文，无需修改
    plt.ylabel('Accuracy')  # Y轴标签已经是英文，无需修改
    plt.legend()  # 因为已经添加了label参数，这里不会出现"No artists with labels found"的警告
    plt.tight_layout()
    plt.savefig('accuracy.png')
    plt.show()

    # 绘制损失曲线
    plt.plot(losses)
    plt.title('DQN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('loss.png')
