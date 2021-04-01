from gym.envs.registration import register

register(
    id='Trading-v0',
    entry_point='trading_env.envs:TradingEnv',
)

register(
    id='Trading-v1',
    entry_point='trading_env.envs:TradingCARG',
)

register(
    id='Trading-v2',
    entry_point='trading_env.envs:TradingAction',
)

register(
    id='Trading-v3',
    entry_point='trading_env.envs:AugmentedEnv',
)

register(
    id='Trading-v4',
    entry_point='trading_env.envs:AppleEnv',
)

register(
    id='Trading-v5',
    entry_point='trading_env.envs:McDonaldEnv',
)


