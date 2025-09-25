import os
import time
import copy
import numpy as np
import torch
import gymnasium as gym
import env  # 触发环境注册
import PPO_model


def _flatten_action(actions, batch_size):
    """把 (3, B) 或其它形状的动作展平成 1D(3B) 以匹配 MultiDiscrete。"""
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    return np.asarray(actions).reshape(-1)


def get_validate_env(env_paras):
    """
    从验证集目录生成并返回验证环境
    目录结构: ./data_dev/{num_jobs}{num_mas:02d}/ 下若干实例文件
    """
    file_path = "./data_dev/{0}{1}/".format(
        env_paras["num_jobs"], str(env_paras["num_mas"]).zfill(2)
    )
    valid_data_files = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    env_inst = gym.make(
        "fjsp-v0",
        case=valid_data_files,          # 用文件列表作为 case
        env_paras=env_paras,
        data_source="file",
        disable_env_checker=True,       # 仍可能有默认 wrapper
    )
    return env_inst


def validate(env_paras, env_inst, model_policy):
    """
    训练过程中的验证，流程与测试类似
    返回: (平均 makespan, 逐实例 makespan 向量)
    """
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))

    # Gymnasium reset
    _obs, _info = env_inst.reset()
    base = env_inst.unwrapped              # <<< 关键：访问原始环境
    state = base.state
    done_flag = False
    dones_vec = base.done_batch

    while not done_flag:
        with torch.no_grad():
            actions = model_policy.act(
                state, memory, dones_vec, flag_sample=False, flag_train=False
            )
        actions_flat = _flatten_action(actions, batch_size)

        _obs, reward_scalar, terminated, truncated, info = env_inst.step(actions_flat)

        state = base.state
        dones_vec = torch.from_numpy(info["done_batch"]).to(env_paras["device"])
        done_flag = bool(terminated) or bool(truncated)

    gantt_result = base.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")

    makespan_mean = copy.deepcopy(base.makespan_batch.mean())
    makespan_batch = copy.deepcopy(base.makespan_batch)

    env_inst.reset()
    mean_val = float(makespan_mean.item())
    #print(f'validating time: {dur:.3f}s | mean makespan over {batch_size} instances = {mean_val:.3f}')
    print('validating time:', time.time() - start, '\n')
    return makespan_mean, makespan_batch

