from sbx import DroQ

model = DroQ('MlpPolicy', env, learning_rate=0.001, learning_starts=10000, gradient_steps=20, policy_delay=20,
            dropout_rate=0.01, layer_norm=True)
