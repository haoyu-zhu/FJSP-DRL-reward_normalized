# test.py —— FJSP 评测脚本（兼容 Gymnasium & 自定义老接口）
# 关键点：
# - 统一使用 env.unwrapped 访问自定义属性（state / done_batch / makespan_batch / validate_gantt）
# - 兼容 step 返回 (state, reward, dones) 或 (obs, reward, terminated, truncated, info)
# - 一次性写 Excel，避免循环 save/close
# - 规则 "DRL" 会自动把 ./model/ 目录的所有 .pt 载入逐个评测
# - 模型权重放在 ./model/ 下；见 “### [模型文件导入处] ###”

import copy
import json
import os
import random
import time as time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import PPO_model
from env.load_data import nums_detec

# 如果你的环境参数用的不是 "files"（比如旧版用 "file"），改这里：
DATA_SOURCE = "files"


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def get_device():
    # 在 ROCm 上，PyTorch 仍把设备名显示为 "cuda"
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _as_bool(x):
    """鲁棒地把各种可能的 done 表示转换成 python bool（支持标量、np 数组、torch 张量）"""
    if isinstance(x, bool):
        return x
    if torch.is_tensor(x):
        # 允许张量形如 [True, True, ...]
        return bool(x.all().item())
    if isinstance(x, (np.ndarray, list, tuple)):
        return bool(np.all(x))
    try:
        # 包含 .all() 的其它类型
        return bool(x.all())
    except Exception:
        return bool(x)


def schedule(env_unwrapped, model, memories, flag_sample=False):
    """
    执行一次调度，返回 (makespan_batch, spend_time)
    env_unwrapped: 必须是 env.unwrapped（裸环境）
    """
    core = env_unwrapped

    # reset 兼容：Gymnasium 返回 (obs, info)，老接口可能返回 obs 或什么都不返回
    try:
        reset_ret = core.reset()
        if isinstance(reset_ret, tuple):
            obs = reset_ret[0]
        else:
            obs = reset_ret
    except Exception:
        obs = None  # 如果 reset 不返回东西也无所谓

    # 初始状态：优先用自定义 core.state；不存在就用 reset 的 obs
    state = getattr(core, "state", None)
    if state is None:
        state = obs

    # 初始 dones：优先用自定义 core.done_batch，否则先用 False
    dones = getattr(core, "done_batch", False)
    done_flag = _as_bool(dones)

    t0 = time.time()
    # 进入循环
    while not done_flag:
        with torch.no_grad():
            actions = model.policy_old.act(
                state, memories, dones, flag_sample=flag_sample, flag_train=False
            )

        step_ret = core.step(actions)

        # 兼容两种 step 签名
        if isinstance(step_ret, tuple) and len(step_ret) == 3:
            # 老接口: (state, rewards, dones)
            state, _, dones = step_ret
            done_flag = _as_bool(dones)
        elif isinstance(step_ret, tuple) and len(step_ret) == 5:
            # Gymnasium: (obs, reward, terminated, truncated, info)
            obs, _, terminated, truncated, _ = step_ret
            # 仍优先从自定义属性拿“状态”，否则用 obs
            state = getattr(core, "state", obs)
            # dones 既给下一步 act 用，也用来判断循环结束
            done_flag = _as_bool(np.array(terminated) | np.array(truncated))
            # 传给策略的 dones（保持类型宽松，策略侧通常只关心 bool 含义）
            dones = done_flag
        else:
            raise RuntimeError(
                f"env.step 返回未知签名，得到 {type(step_ret)} 长度={len(step_ret) if isinstance(step_ret, tuple) else 'NA'}"
            )

    spend_time = time.time() - t0

    # 校验甘特合法（如果环境实现了）
    try:
        ok = core.validate_gantt()[0]
        if not ok:
            print("Scheduling Error！！！！！！")
    except Exception:
        pass

    # 兼容 makespan_batch 可能是张量或标量
    return copy.deepcopy(getattr(core, "makespan_batch", None)), spend_time


def main():
    setup_seed(2023)
    device = get_device()
    print("PyTorch device:", device.type)
    torch.set_printoptions(threshold=np.inf, linewidth=120, sci_mode=False)

    # 读取配置
    with open("./config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    env_paras = cfg["env_paras"]
    model_paras = cfg["model_paras"]
    train_paras = cfg["train_paras"]
    test_paras = cfg["test_paras"]

    # 注入设备
    env_paras["device"] = device
    model_paras["device"] = device

    # 计算网络输入维度（保持你原来的写法）
    model_paras["actor_in_dim"] = (
        model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    )
    model_paras["critic_in_dim"] = (
        model_paras["out_size_ma"] + model_paras["out_size_ope"]
    )

    # 测试集文件
    data_path = Path("./data_test") / test_paras["data_path"]
    all_files = sorted(os.listdir(str(data_path)), key=lambda x: x[:-4])
    num_ins = int(test_paras["num_ins"])
    test_files = all_files[:num_ins]

    # 规则/模型列表
    rules = list(test_paras["rules"])
    model_dir = Path("./model")
    model_files = [f for f in os.listdir(str(model_dir)) if f.endswith(".pt")]

    # 如果包含 "DRL"，把 ./model/ 下所有 .pt 加进来
    if "DRL" in rules:
        for f in model_files:
            if f not in rules:
                rules.append(f)
        rules = [r for r in rules if r != "DRL"]

    # 初始化 PPO
    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)

    # 输出目录
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = Path("./save") / f"test_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 结果表头（文件名列）
    all_makespan_cols = {"file_name": [*test_files]}
    all_time_cols = {"file_name": [*test_files]}

    # 预创建/复用 env
    envs = [None] * num_ins

    start_all = time.time()
    for rule_idx, rule in enumerate(rules):
        print("\n=== rule:", rule, "===")

        # ------------------------------------------------------------
        # ### [模型文件导入处] ###
        # 如果 rule 是 ".pt" 文件，就从 ./model/<rule> 加载权重
        if rule.endswith(".pt"):
            ckpt_path = model_dir / rule
            if device.type == "cuda":
                state_dict = torch.load(str(ckpt_path))
            else:
                state_dict = torch.load(str(ckpt_path), map_location="cpu")
            model.policy.load_state_dict(state_dict)
            model.policy_old.load_state_dict(state_dict)
            print("Loaded checkpoint:", rule)
        # ------------------------------------------------------------

        # 每条 rule 下的 env 批大小
        env_test_paras = copy.deepcopy(env_paras)
        if test_paras.get("sample", False):
            env_test_paras["batch_size"] = int(test_paras["num_sample"])
        else:
            env_test_paras["batch_size"] = 1

        per_rule_makespans = []
        per_rule_times = []

        t_rule = time.time()
        for i_ins in range(num_ins):
            test_file = str(data_path / test_files[i_ins])
            with open(test_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            ins_num_jobs, ins_num_mas, _ = nums_detec(lines)
            env_test_paras["num_jobs"] = int(ins_num_jobs)
            env_test_paras["num_mas"] = int(ins_num_mas)

            # 创建或复用 env（用 unwrapped）
            if envs[i_ins] is None:
                if test_paras.get("sample", False):
                    raw = gym.make(
                        "fjsp-v0",
                        case=[test_file] * int(test_paras["num_sample"]),
                        env_paras=env_test_paras,
                        data_source=DATA_SOURCE,
                        disable_env_checker=True,
                    )
                else:
                    raw = gym.make(
                        "fjsp-v0",
                        case=[test_file],
                        env_paras=env_test_paras,
                        data_source=DATA_SOURCE,
                        disable_env_checker=True,
                    )
                env = raw.unwrapped
                envs[i_ins] = env
                print(f"Create env[{i_ins}] for: {test_files[i_ins]}")
            else:
                env = envs[i_ins]

            # 评测
            if test_paras.get("sample", False):
                makespan, spend = schedule(env, model, memories, flag_sample=True)
                if torch.is_tensor(makespan):
                    per_rule_makespans.append(float(torch.min(makespan).item()))
                else:
                    per_rule_makespans.append(float(makespan))
                per_rule_times.append(float(spend))
            else:
                runs = int(test_paras.get("num_average", 1))
                ms_buf, tm_buf = [], []
                for _ in range(runs):
                    makespan, spend = schedule(env, model, memories, flag_sample=False)
                    if torch.is_tensor(makespan):
                        val = makespan.item() if makespan.ndim == 0 else makespan.mean().item()
                    else:
                        val = float(makespan)
                    ms_buf.append(val)
                    tm_buf.append(float(spend))
                    try:
                        env.reset()
                    except Exception:
                        pass
                per_rule_makespans.append(float(np.mean(ms_buf)))
                per_rule_times.append(float(np.mean(tm_buf)))

            print(f"finish env {i_ins}")

        print("rule_spend_time:", time.time() - t_rule, "sec")

        # 收集列
        all_makespan_cols[rule] = per_rule_makespans
        all_time_cols[rule] = per_rule_times

        # 评估完，所有 env 复位
        for e in envs:
            if e is not None:
                try:
                    e.reset()
                except Exception:
                    pass

    print("total_spend_time:", time.time() - start_all, "sec")

    # 一次性写 Excel
    makespan_df = pd.DataFrame(all_makespan_cols)
    time_df = pd.DataFrame(all_time_cols)

    makespan_xlsx = save_dir / f"makespan_{ts}.xlsx"
    time_xlsx = save_dir / f"time_{ts}.xlsx"

    with pd.ExcelWriter(makespan_xlsx, mode="w") as w:
        makespan_df.to_excel(w, sheet_name="Sheet1", index=False)
    with pd.ExcelWriter(time_xlsx, mode="w") as w:
        time_df.to_excel(w, sheet_name="Sheet1", index=False)

    print("Saved:", makespan_xlsx)
    print("Saved:", time_xlsx)


if __name__ == "__main__":
    main()
