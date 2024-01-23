modules = {
    'connectx_zero.py',
    'connectx.py',
    'mcts.py',
    'nnet_torch.py'
}
imports = """
from torch import tensor
from collections import OrderedDict
"""
agent = """
connectx_agent = None
initialized = False
def agent(obs, config):
    global connectx_agent
    global initialized
    if not initialized:
        connectx_agent = ConnectXAgent(config, device='cuda')
        connectx_agent.load_policy(state_dict)
        initialized = True
    s = obs['board']
    mark = obs['mark']
    move = int(connectx_agent.choose_move(s, mark))
    valid_moves = legal_moves(np.reshape(s, (config.rows, config.columns)), config)
    if move not in valid_moves:
        valid = valid_moves[0]  
        move = valid
    return move
"""

def write_agent(filepath, state_dict):
    with open(filepath, 'w', encoding="utf-8") as agent_file:
        agent_file.write(imports + "\n")
        for module in modules:
            with open(module) as f:
                agent_file.write(f.read() + "\n\n")
        agent_file.write("state_dict = " + str(state_dict) + "\n\n")
        agent_file.write(agent)