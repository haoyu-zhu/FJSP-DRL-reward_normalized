import copy
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import gymnasium as gym
import env  # 触发 env.__init__ 中的注册（很重要）
import pandas as pd

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _flatten_action(actions, batch_size):
    """把 (3, B) 或其它形状的动作展平成 1D(3B)，匹配 Gymnasium 的 MultiDiscrete。"""
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    return np.asarray(actions, dtype=np.int64).reshape(-1)


def main():
    # ---------- PyTorch 初始化（兼容 PyTorch 2.1+） ----------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch device:", device.type)

    torch.set_default_dtype(torch.float32)
    try:
        torch.set_default_device(device)  # PyTorch 2.1+
    except AttributeError:
        if device.type == "cuda":
            torch.cuda.set_device(device)

    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None,
                           linewidth=None, profile=None, sci_mode=False)

    # ---------- 读取配置 ----------
    with open("./config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_paras = cfg["env_paras"]
    model_paras = cfg["model_paras"]
    train_paras = cfg["train_paras"]

    env_paras["device"] = device
    model_paras["device"] = device

    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]

    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    # ---------- PPO & 验证环境 ----------
    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    env_valid = get_validate_env(env_valid_paras)  # 验证环境（从文件加载）

    # ---------- 保存目录 ----------
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = f'./save/train_{str_time}'
    os.makedirs(save_path, exist_ok=True)

    # 记录验证曲线数据
    val_iters = []
    valid_results = []          # 每次验证的平均 makespan
    valid_results_100 = []      # 每次验证的逐实例 makespan（向量）

    # 保存最优模型
    best_models = deque()
    maxlen = 1
    makespan_best = float('inf')

    # ---------- 训练 ----------
    start_time = time.time()
    env_inst = None
    base = None  # 原始环境（unwrapped）

    for i in range(1, train_paras["max_iterations"] + 1):
        # 每 parallel_iter 次替换一批训练实例
        if (i - 1) % train_paras["parallel_iter"] == 0:
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)

            env_inst = gym.make(
                'fjsp-v0',
                case=case,
                env_paras=env_paras,
                data_source="case",
                disable_env_checker=True,   # 阻止 env_checker，但仍可能有默认 wrapper
            )
            base = env_inst.unwrapped     # <<< 关键：拿到你自定义的原始环境
            print('num_job:', num_jobs, '\tnum_mas:', num_mas, '\tnum_opes:', sum(nums_ope))
            _obs, _info = env_inst.reset()  # Gymnasium: (obs, info)

        # 调度一个完整 episode
        done_flag = False
        dones_vec = base.done_batch       # 逐并行实例完成标记 (B,)
        state = base.state                # 仍然把 EnvState 传给策略

        t0 = time.time()
        while not done_flag:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones_vec)

            # (3,B) -> (3B,)
            actions_flat = _flatten_action(actions, env_paras["batch_size"])

            # Gymnasium 的五元组
            _obs, reward_scalar, terminated, truncated, info = env_inst.step(actions_flat)

            # 策略输入仍用 base.state（与现有 PPO 兼容）
            state = base.state

            # 从 info 取逐实例 reward/done，喂进 Memory（你的 PPO 是多环境并行）
            reward_batch = torch.from_numpy(info["reward_batch"]).to(device)   # (B,)
            dones_vec = torch.from_numpy(info["done_batch"]).to(device)        # (B,)
            memories.rewards.append(reward_batch)
            memories.is_terminals.append(dones_vec)

            done_flag = bool(terminated) or bool(truncated)

        print("spend_time:", time.time() - t0)

        # 验证调度可行性（在原始环境上）
        gantt_ok = base.validate_gantt()[0]
        if not gantt_ok:
            print("Scheduling Error！！！！！！")

        # 重置环境，开始下一个 episode
        env_inst.reset()

        # ---- 按频率更新策略 ----
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print(f"reward: {reward:.3f}; loss: {loss:.3f}")
            memories.clear_memory()

        # ---- 按频率进行验证 & 保存最优 ----
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            vali_mean, vali_vec = validate(env_valid_paras, env_valid, model.policy_old)
            #vali_mean, vali_vec = validate(env_valid_paras, env_valid, model.policy_old)

            vm = float(vali_mean.item() if torch.is_tensor(vali_mean) else vali_mean)
            print(f'[valid] iter {i}: mean makespan across 100 test instances = {vm:.3f}')

            val_iters.append(i)
            valid_results.append(float(vali_mean.item() if torch.is_tensor(vali_mean) else vali_mean))
            if torch.is_tensor(vali_vec):
                vali_vec = vali_vec.detach().cpu().numpy()
            else:
                vali_vec = np.asarray(vali_vec)
            valid_results_100.append(vali_vec)

            # 保存最优
            if valid_results[-1] < makespan_best:
                makespan_best = valid_results[-1]
                if len(best_models) == maxlen:
                    try:
                        delete_file = best_models.popleft()
                        os.remove(delete_file)
                    except Exception:
                        pass
                save_file = f'{save_path}/save_best_{num_jobs}_{num_mas}_{i}.pt'
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

    # ---------- 训练结束：写出验证曲线到 Excel ----------
    # 1) 平均曲线
    df_ave = pd.DataFrame({
        "iterations": val_iters,
        "res": valid_results
    })
    df_ave.to_excel(f'{save_path}/training_ave_{str_time}.xlsx', index=False)

    # 2) 逐实例（列名自动编号）
    if len(valid_results_100) > 0:
        arr = np.vstack(valid_results_100)  # shape: (num_validations, batch_size)
        df100 = pd.DataFrame(arr, columns=[f"inst_{k}" for k in range(arr.shape[1])])
        df100.insert(0, "iterations", val_iters)
        df100.to_excel(f'{save_path}/training_100_{str_time}.xlsx', index=False)

    print("total_time:", time.time() - start_time)


if __name__ == "__main__":
    main()