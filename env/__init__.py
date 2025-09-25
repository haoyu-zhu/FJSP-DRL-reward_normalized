# env/__init__.py
from gymnasium.envs.registration import register

# 避免重复注册：多次导入时如果已存在就跳过
try:
    register(
        id="fjsp-v0",
        entry_point="env.fjsp_env:FJSPEnv",  # 包.模块:类
        # 可选：如果需要 TimeLimit 包裹可设置，例如：
        # max_episode_steps=10_000,
    )
except Exception as e:
    # 已注册就忽略（Gymnasium 会在重复注册时抛错）
    if "already registered" not in str(e).lower():
        raise
