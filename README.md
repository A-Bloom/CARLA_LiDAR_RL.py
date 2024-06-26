## Using RL and 2D LiDAR observations to control a car and avoid obstacles in CARLA.

The BackEnv contains top level functions that all the other environments need to function.<br>
All the other environments inherit the BackEnv function and add specific Action Spaces, Observation Spaces, and Reward Functions

This table lists the various features of each environment:
<br> All Environments give -1 reward for crashing.

| Environment | Action Space                                                                                        | Observation Space                                                    | Reward Function                                                                         |
|-------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Env1        | 21 discrete throttle actions from 1 reverse to 1 forward<br> 21 discrete steer actions from -1 to 1 | Blank grid with ones signifying obstacles<br> Precision of 1/4 meter | Rewards for going faster up to 50 m/s with a discount for turning<br> Max reward of 0.5 |
| Env2        | 21 discrete steer actions from -1 to 1                                                              | Blank grid with ones signifying obstacles<br> Precision of 1/4 meter | Reward for throttle, throttle is decreased proportionally to steer                      |
| Env3        |                                                                                                     |                                                                      |                                                                                         |
| Env4        |                                                                                                     |                                                                      |                                                                                         |
| Env5        |                                                                                                     |                                                                      |                                                                                         |
| Env6        |                                                                                                     |                                                                      |                                                                                         |
| Env7        |                                                                                                     |                                                                      |                                                                                         |
| Env8        |                                                                                                     |                                                                      |                                                                                         |

The SuperTrainer is used to test multiple environments with multiple hyperparameters and then graph them against performance.