from gym.envs.registration import register

register(
    id='Trading-v0',
    entry_point='trading_env.envs:TradingEnv',
)


register(
    id='Trading-v1',
    entry_point='trading_env.envs:TradingEnv_State',
)

register(
    id='Trading-v2',
    entry_point='trading_env.envs:TradingEnv_State2',
)

register(
    id='Trading-v3',
    entry_point='trading_env.envs:MCDEnv',
)


register(
    id='Trading-v4',
    entry_point='trading_env.envs:TradingEnvF',
)



# register(
#     id='Trading-v1',
#     entry_point='trading_env.envs:TradingEnvFix',
# )

# register(
#     id='Trading-v2',
#     entry_point='trading_env.envs:NormalizedEnv',
# )
# <<<<<<< HEAD

# register(
#     id='Trading-v3',
#     entry_point='trading_env.envs:AugmentedEnv',
# )

# register(
#     id='Trading-v4',
#     entry_point='trading_env.envs:AppleEnv',
# )
# <<<<<<< HEAD

# register(
#     id='Trading-v5',
#     entry_point='trading_env.envs:McDonaldEnv',
# )


# =======
# >>>>>>> parent of b420023... commit before refactoring archi
# =======
# >>>>>>> parent of 69367ab... Stupid reset
