AI Journey: Understanding zero_grad() in PyTorch
Part of my ongoing AI/ML learning journey, this repo explores a subtle but important PyTorch concept: gradient accumulation.
What's inside

A basic training loop using nn.Linear to fit a simple shoe-size-to-height model
Two versions of the same loop: one with optimizer.zero_grad(), one without
Output comparisons showing how omitting zero_grad() doesn't crash the program, it just silently produces unstable, poorly-converging training

Key takeaway
PyTorch accumulates gradients by default. Without clearing them each round, adjustments compound and training becomes unstable, a bug that's easy to miss because nothing throws an error.
Stack
Python PyTorch nn.Linear SGD Optimizer
