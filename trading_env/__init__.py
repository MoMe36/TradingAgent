from gym.envs.registration import register

register(
    id='Trading-v0',
    entry_point='trading_env.envs:TradingEnv',
)

register(
    id='Trading-v1',
    entry_point='trading_env.envs:TradingEnvFix',
)

register(
    id='Trading-v2',
    entry_point='trading_env.envs:NormalizedEnv',
)
