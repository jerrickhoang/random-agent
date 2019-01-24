# A study of the random baseline on RL benchmarks

This short experiment studies the results of running a random baseline on various gym RL benchmarks. The random baseline is defined to be one where the agent acts independently random at every timestep.

### The CartPole-v0 environment

TODO(jhoang): short animation of the environment

#### Description:
A pole is attached by an un-actuated joint to a cart, which moves along frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.


    Observation:
        Type: Box(4)
        Num	Observation              Min            Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        Note: The amount the velocity is reduced or increased is not fixed as it
        depends on the angle the pole is pointing. This is because the center of
        gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

We run 500 episodes and plot the cumulative reward. The following graph shows all 500 runs along with the maximum and the average line.
![](rnd/experiments/CartPole-v0_500/cumulative_reward.png)

We can see that the average for 500 runs is about 22
