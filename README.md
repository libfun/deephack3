## Public repo with Will Press Lever For Food solution at DeepHack.RL

[rl.deephack.me](http://rl.deephack.me)

Gym submissions:
- [Skiing-v0](https://gym.openai.com/evaluations/eval_1ezBtrPT2WCCOSGZt1cwA) -3492.77 ± 1.94
- [MsPacman-v0](https://gym.openai.com/evaluations/eval_6WIeLKoZSWywYDsiq91A) 5512.30 ± 57.51
- [Centipede-v0](https://gym.openai.com/evaluations/eval_gTPovWiQWupVHhDjBQ) 7511.87 ± 445.89

Team Will Press Lever For Food:

[Alexander Guschin](https://github.com/aguschin), [Sergey Korolev](https://github.com/libfun), [Sergey Ovcharenko](https://github.com/dudevil), [Sergey Sviridov](https://github.com/ssviridov), [Max Kharchenko](https://github.com/2sick2speak)

## How to reproduce results
1. `requirements.txt`
2. `python tensorpack/setup.py install` to install modified tensorpack version.
3. Train modified universe A3C agent (https://github.com/libfun/deephack3/blob/master/universe-starter-agent/README.md)
4. Train modified tensorpack A3C agent (https://github.com/libfun/deephack3/blob/master/tensorpack/examples/A3C-Gym/README.md)
5. `player.py` contains code for agent combination. Examples in `Untitled.ipynb`.
