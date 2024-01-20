from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from connectx import ConnectFourGym
from connectx import get_win_percentages, build_agent
from custom_cnn import CustomCNN

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super.__init(self, verbose)

    def _on_step(self) -> bool:
        eval_agent = build_agent(self.model)
        win_rate = get_win_percentages(agent1=eval_agent, agent2="random", n_rounds=10)
        self.logger.record("win_rate", win_rate)
        return True

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)

# Initialize agent
env = ConnectFourGym(agent2="random")
model = DQN(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log="./tensorboard/",
)

# Train agent
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("models/dqn_test")