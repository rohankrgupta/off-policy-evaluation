# Learning the BPS target cheaply, and reusing it (CartPole, discrete actions)

Quality = Var_q(OIS)/Var(g) on 1e6 fresh episodes; 1.0 = on-policy, lower is better.

## Offline fit by pilot size (force noise 0)

The output layer of q is fitted to the pilot's importance-sampling estimate of the variance with a KL pull toward pi_e (weight 10); 2 pilots per size.

| target | quality (mean of 2) | worst | fit time |
|---|---|---|---|
| offline fit, 256 pilot episodes | 1.147 | 1.292 | 2 s |
| offline fit, 512 pilot episodes | 0.493 | 0.542 | 5 s |
| offline fit, 1024 pilot episodes | 0.388 | 0.454 | 12 s |
| offline fit, 2048 pilot episodes | 0.272 | 0.277 | 17 s |
| BPG target, 3.2e6 episodes (main experiment) | 0.257 | – | 47 s |

## Transfer across force-noise levels

0 N: v = 35.998, Var(g) = 268.1, 3 N: v = 34.810, Var(g) = 262.1, 6 N: v = 31.813, Var(g) = 241.9.

| target | used at 0 N | used at 3 N | used at 6 N |
|---|---|---|---|
| BPG, learned at 0 N | 0.257 | 0.329 | 0.497 |
| BPG, learned at 3 N | 0.265 | 0.331 | 0.487 |
| offline fit, 2048 pilot episodes at 0 N | 0.268 | 0.337 | 0.499 |
| offline fit, 2048 pilot episodes at 3 N | 0.288 | 0.352 | 0.502 |
| BPG, learned natively at each level | 0.257 | 0.331 | – |
