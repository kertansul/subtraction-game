import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from src import agents
from tfrl import environment, network


# init environment
env = tf_py_environment.TFPyEnvironment(
    environment.TwoPlayerSubtractionEnv(
        agents.Minimax(max_depth=1), min_value=5, max_value=6
    )
)
qnet = network.QNetwork(
    env.observation_spec(),
    env.action_spec(),
)
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=qnet,
    optimizer=tf.compat.v1.train.AdamOptimizer(0.0003))

replay_buffer_capacity = 20000
collect_episodes_per_iteration = 10

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=replay_buffer_capacity)
# Add an observer that adds to the replay buffer:
replay_observer = [replay_buffer.add_batch]
collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
  env,
  agent.collect_policy,
  observers=replay_observer,
  num_episodes=collect_episodes_per_iteration)

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=1024,
    num_steps=2).prefetch(3)
iterator = iter(dataset)


# Reset the train step.
agent.train_step_counter.assign(0)
# Reset the environment.
time_step = env.reset()


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

num_eval_episodes = 100
eval_env = tf_py_environment.TFPyEnvironment(
    environment.TwoPlayerSubtractionEnv(
        agents.Minimax(max_depth=1), min_value=5, max_value=6)
)


log_interval = 10
eval_interval = 500
save_interval = 500


for _ in range(15000):

    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))

    if step % save_interval == 0:
        qnet.save_weights('/vol/08822801/shawn.chen/subtraction_game/tfrl/test001', step=step)