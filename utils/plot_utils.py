import matplotlib.pyplot as plt

def update_plot(rewards):
    fig, ax = plt.subplots()
    ax.plot(rewards, label="Episode Reward")
    ax.set_title("Training Reward Progress")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.savefig("temp_plot.png")
    plt.close(fig)

