import sys
import gymnasium as gym
from gymnasium import spaces
import torch
from dataclasses import dataclass
from env.load_data import load_fjs, nums_detec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
from utils.my_utils import read_json, write_json


@dataclass
class EnvState:
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None
    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None
    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch
        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    return feat_job_batch.gather(1, opes_appertain_batch)


class FJSPEnv(gym.Env):
    """
    Gymnasium-compatible FJSP environment (batched inside).
    - Actions: MultiDiscrete(flattened [opes]*B + [mas]*B + [jobs]*B)
    - Observation: Dict of fixed-shape float32 arrays
    - Reward: mean over batch (scalar). Per-batch rewards in info["reward_batch"].
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, case, env_paras, data_source='case', render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ---- load paras ----
        self.show_mode = env_paras["show_mode"]
        self.batch_size = env_paras["batch_size"]
        self.num_jobs = env_paras["num_jobs"]
        self.num_mas = env_paras["num_mas"]
        self.paras = env_paras
        self.device = env_paras["device"]

        # ---- load instances ----
        num_data = 8
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        if data_source == 'case':
            for i in range(self.batch_size):
                lines.append(case.get_case(i)[0])
                _, _, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        else:
            for i in range(self.batch_size):
                with open(case[i]) as f:
                    line = f.readlines()
                    lines.append(line)
                _, _, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)

        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], self.num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        # ---- dynamic feats ----
        self.proc_times_batch = torch.stack(tensors[0], dim=0).to(self.device)       # (B, O, M)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long().to(self.device)  # (B, O, M)
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float().to(self.device)  # (B, O, O)

        # ---- static feats ----
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0).to(self.device)   # (B, O, O)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0).to(self.device)   # (B, O, O)
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long().to(self.device)  # (B, O)
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long().to(self.device)  # (B, J)
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long().to(self.device)       # (B, J)
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1   # (B, J)
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1).to(self.device)            # (B,)

        # ---- dynamic variables ----
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)
        self.time = torch.zeros(self.batch_size, device=self.device)
        self.N = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)

        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)  # (B, J)

        # raw features
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes), device=self.device)
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas), device=self.device)

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze(1)

        end_time_batch = (feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)

        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1).float()

        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # masks
        self.mask_job_procing_batch = torch.zeros((self.batch_size, self.num_jobs), dtype=torch.bool, device=self.device)
        self.mask_job_finish_batch = torch.zeros((self.batch_size, self.num_jobs), dtype=torch.bool, device=self.device)
        self.mask_ma_procing_batch = torch.zeros((self.batch_size, self.num_mas), dtype=torch.bool, device=self.device)

        # partial schedules
        self.schedules_batch = torch.zeros((self.batch_size, self.num_opes, 4), device=self.device)
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]

        # machine states
        self.machines_batch = torch.zeros((self.batch_size, self.num_mas, 4), device=self.device)
        self.machines_batch[:, :, 0] = 1.0  # idle

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        self.state = EnvState(
            batch_idxes=self.batch_idxes, feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
            proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
            ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
            mask_job_procing_batch=self.mask_job_procing_batch, mask_job_finish_batch=self.mask_job_finish_batch,
            mask_ma_procing_batch=self.mask_ma_procing_batch, opes_appertain_batch=self.opes_appertain_batch,
            ope_step_batch=self.ope_step_batch, end_ope_biases_batch=self.end_ope_biases_batch,
            time_batch=self.time, nums_opes_batch=self.nums_opes
        )

        # keep copies for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

        # --------- Gymnasium spaces ----------
        # actions: flattened [opes]*B + [mas]*B + [jobs]*B
        nvec = np.concatenate([
            np.full(self.batch_size, self.num_opes, dtype=np.int64),
            np.full(self.batch_size, self.num_mas, dtype=np.int64),
            np.full(self.batch_size, self.num_jobs, dtype=np.int64),
        ])
        self.action_space = spaces.MultiDiscrete(nvec)

        # observations (all float32; bool/int cast to float32/0-1)
        def f32_box(shape, low=-np.inf, high=np.inf):
            return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "feat_opes":       f32_box((self.batch_size, self.paras["ope_feat_dim"], self.num_opes)),
            "feat_mas":        f32_box((self.batch_size, self.paras["ma_feat_dim"], self.num_mas)),
            "proc_times":      f32_box((self.batch_size, self.num_opes, self.num_mas), low=0.0, high=np.inf),
            "ope_ma_adj":      f32_box((self.batch_size, self.num_opes, self.num_mas), low=0.0, high=1.0),
            "mask_job_proc":   f32_box((self.batch_size, self.num_jobs), low=0.0, high=1.0),
            "mask_job_finish": f32_box((self.batch_size, self.num_jobs), low=0.0, high=1.0),
            "mask_ma_proc":    f32_box((self.batch_size, self.num_mas), low=0.0, high=1.0),
            "ope_step":        f32_box((self.batch_size, self.num_jobs), low=0.0, high=float(self.num_opes)),
            "time":            f32_box((self.batch_size,), low=0.0, high=np.inf),
        })

    # -------- helper to build obs --------
    def _obs(self):
        to_np = lambda t: t.detach().float().cpu().numpy().astype(np.float32)
        return {
            "feat_opes":       to_np(self.feat_opes_batch),
            "feat_mas":        to_np(self.feat_mas_batch),
            "proc_times":      to_np(self.proc_times_batch),
            "ope_ma_adj":      to_np(self.ope_ma_adj_batch.float()),
            "mask_job_proc":   to_np(self.mask_job_procing_batch.float()),
            "mask_job_finish": to_np(self.mask_job_finish_batch.float()),
            "mask_ma_proc":    to_np(self.mask_ma_procing_batch.float()),
            "ope_step":        to_np(self.ope_step_batch.float()),
            "time":            to_np(self.time),
        }

    # ------------- step -------------
    def step(self, action):
        """
        action: MultiDiscrete vector of length 3*B:
            [ope_0..ope_{B-1}, ma_0..ma_{B-1}, job_0..job_{B-1}]
        """
        # a = np.asarray(action, dtype=np.int64).reshape(3, self.batch_size)
        # actions = torch.from_numpy(a).to(self.device)
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.size % 3 != 0:
            raise ValueError(f"Action length must be a multiple of 3, got {arr.size}")

        k = arr.size // 3
        act = arr.reshape(3, k)

        active = int(self.batch_idxes.numel())
        if k == active:
            # 正常：仅活跃实例动作
            act_used = act
        elif k == self.batch_size:
            # 也接受全量动作，按 batch_idxes 挑选活跃部分
            idx = self.batch_idxes.detach().cpu().numpy()
            act_used = act[:, idx]
        else:
            raise ValueError(
                f"Unexpected action length: got {k}, expected |batch_idxes|={active} or batch_size={self.batch_size}"
            )

        actions = torch.as_tensor(act_used, device=self.device, dtype=torch.long)

        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        # remove unselected O-M arcs
        remain_ope_ma_adj = torch.zeros((self.batch_size, self.num_mas), dtype=torch.int64, device=self.device)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # update selected opes features
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack((
            torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
            torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
            proc_times.float()), dim=1)

        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs],
                                self.num_opes - 1, opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # number of unscheduled ops in the job
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1

        # update start time & job completion time
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled
        un_scheduled = 1 - is_scheduled
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze(1) * un_scheduled
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times

        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] + self.feat_opes_batch[self.batch_idxes, 2, :]) \
            .gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(
            end_time_batch, self.opes_appertain_batch[self.batch_idxes, :])

        # update partial schedule / machines
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), device=self.device), mas.float()), dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = 0.0
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # machine features
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(
            self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # flags
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 torch.ones_like(self.mask_job_finish_batch, dtype=torch.bool),
                                                 self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        # reward (maximize makespan reduction)
        max_end = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max_end
        self.makespan_batch = max_end

        # auto-advance time if no eligible O-M pairs
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # shrink batch_idxes used for scheduling but keep obs shapes fixed
        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size, device=self.device)[mask_finish]

        # update state holder (kept for compatibility)
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
                          self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch,
                          self.mask_ma_procing_batch, self.ope_step_batch, self.time)

        obs = self._obs()
        # Gymnasium scalar reward; batch rewards go to info
        reward_scalar = float(self.reward_batch.mean().item())
        terminated = bool(self.done_batch.all().item())
        truncated = False
        info = {
            "reward_batch": self.reward_batch.detach().cpu().numpy().astype(np.float32),
            "done_batch": self.done_batch.detach().cpu().numpy().astype(bool),
        }
        return obs, reward_scalar, terminated, truncated, info

    def if_no_eligible(self):
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(
            1, ope_step_batch.unsqueeze(-1).expand(-1, -1, self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(op_proc_time)
        flag_trans_2_next_time = torch.sum(
            torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2), dim=[1, 2]
        )
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)
        a = self.machines_batch[:, :, 1]
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        c = torch.min(b, dim=1)[0]
        d = (a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None]
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # set those machines idle
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1.0
        self.machines_batch = aa.transpose(1, 2)

        # update utilization
        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        # FIX: use torch.nonzero instead of np.argwhere(...).to(self.device)
        jobs_index = torch.nonzero(jobs >= 0, as_tuple=False)
        if jobs_index.numel() > 0:
            batch_idxes = jobs_index[:, 0]
            job_idxes = jobs[jobs_index[:, 0], jobs_index[:, 1]].long()
            self.mask_job_procing_batch[batch_idxes, job_idxes] = False
            self.mask_ma_procing_batch[d] = False
            self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                     torch.ones_like(self.mask_job_finish_batch, dtype=torch.bool),
                                                     self.mask_job_finish_batch)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # restore
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size, device=self.device)
        self.time = torch.zeros(self.batch_size, device=self.device)
        self.N = torch.zeros(self.batch_size, device=self.device)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.zeros((self.batch_size, self.num_jobs), dtype=torch.bool, device=self.device)
        self.mask_job_finish_batch = torch.zeros((self.batch_size, self.num_jobs), dtype=torch.bool, device=self.device)
        self.mask_ma_procing_batch = torch.zeros((self.batch_size, self.num_mas), dtype=torch.bool, device=self.device)
        self.schedules_batch = torch.zeros((self.batch_size, self.num_opes, 4), device=self.device)
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros((self.batch_size, self.num_mas, 4), device=self.device)
        self.machines_batch[:, :, 0] = 1.0

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        info = {}
        return self._obs(), info

    # render()/validate_gantt()/get_idx() 与你原先一致（略微小清理）
    def render(self):
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_mas
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)])
                          for _ in range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                fig = plt.figure(figsize=(10, 6))
                fig.canvas.manager.set_window_title('Visual_gantt')
                axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                y_ticks = [f"Machine {i}" for i in range(num_mas)]
                y_ticks_loc = list(range(num_mas, 0, -1))
                labels = [f"job {j+1}" for j in range(num_jobs)]
                patches = [mpatches.Patch(color=color[k], label=labels[k]) for k in range(self.num_jobs)]
                axes.cla()
                axes.set_title('FJSP Schedule')
                axes.grid(linestyle='-.', color='gray', alpha=0.2)
                axes.set_xlabel('Time')
                axes.set_ylabel('Machine')
                axes.set_yticks(y_ticks_loc, y_ticks)
                axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0))
                axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    idx_job, _ = self.get_idx(id_ope, batch_id)
                    id_machine = schedules[id_ope][1]
                    axes.barh(id_machine, 0.2, left=schedules[id_ope][2], color='#b2b2b2', height=0.5)
                    axes.barh(id_machine, schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                              left=schedules[id_ope][2] + 0.2, color=color[idx_job], height=0.5)
                plt.show()
        return

    def get_idx(self, id_ope, batch_id):
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i]) - 1):
                        break
                    if ma_gantt[i][j][2] > ma_gantt[i][j + 1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2] - ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i] + j]
                    step_next = schedule[num_ope_biases[i] + j + 1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0] == 1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass
