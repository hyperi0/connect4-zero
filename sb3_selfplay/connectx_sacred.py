from stable_baselines3 import DQN
from connectx import ConnectFourGym
from custom_cnn import CustomCNN
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("connectx_dqn")
ex.observers.append(MongoObserver(url='localhost:27017', db_name='connectx'))

@ex.config
def cfg():
    agent2 = "random"
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
    )
    verbose = 1
    total_timesteps = 1_000
    log_interval = 4
    save_dir = "models/"

@ex.capture
def make_env(agent2):
    return ConnectFourGym(agent2=agent2)

@ex.capture
def make_model(env, policy_kwargs, verbose):
    return DQN("CnnPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs)

@ex.capture
def train_model(model, total_timesteps, log_interval):
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    return model

@ex.automain
def run(save_dir):
    env = make_env()
    model = make_model(env)
    model = train_model(model)
    model.save(save_dir + "dqn_test")