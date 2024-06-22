from marllib import marl
import gym
import rware

# env = marl.make_env(environment_name="rware", map_name="default_map")
env = marl.make_env(environment_name="rware", map_name="customized_map", n_agents=4, map_size="tiny", force_coop=True)
# env = marl.make_env(environment_name="voltage", map_name="case33_3min_final")

# ia2c_algo.fit(env, ia2c_agent, stop={"timesteps_total": 10000}, local_mode=True, num_gpus=1,
#               num_workers=4, share_policy='group', checkpoint_freq=100)

# # IA2C
# algo = marl.algos.ia2c(hyperparam_source="common")
# agent = marl.build_model(env, algo, {"core_arch":"mlp"})
# algo.fit(env, agent, local_mode=True, num_gpus=1,
#               num_workers=4, checkpoint_freq=100)

# MAA2C
algo = marl.algos.maa2c(hyperparam_source="common")
agent = marl.build_model(env, algo, {"core_arch":"mlp"})
algo.fit(env, agent, local_mode=True, num_gpus=1, stop={"timesteps_total": 2000000},
              num_workers=4, checkpoint_freq=100)

# # COMA
# algo = marl.algos.coma(hyperparam_source="common")
# agent = marl.build_model(env, algo, {"core_arch":"mlp"})
# algo.fit(env, agent, local_mode=True, num_gpus=1,
#               num_workers=4, share_policy='group', checkpoint_freq=100)

# # VDA2C
# algo = marl.algos.vda2c(hyperparam_source="common")
# agent = marl.build_model(env, algo, {"core_arch":"mlp"})
# algo.fit(env, agent, local_mode=True, num_gpus=1,
#               num_workers=4, share_policy='group', checkpoint_freq=100)