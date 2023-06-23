# ProbTrojan
A probabilistic trojaning (backdoor) attack on deep networks.

It is based on version 1.0.8 of the [Trojanzoo](https://github.com/ain-soph/trojanzoo) framework,
which has been forked for this project.
For detailed information on how to execute attacks and defenses, please refer to the Trojanzoo documentation.

# The probabilistic attack

The idea of this attack is to use more than one trigger in the training phase,
intentionally keeping the ASR (Attack Success Ratio) low for each trigger.
This is supposed to make the attack stealthier and more evasive.
At test time, the attacker tries the triggers one by one, until hopefully one of them works.

# How to use
Trojanzoo has some [examples](./examples) of running attacks and defenses.
Specifically, see the attack example. It uses a CLI command with some arguments specifying parameters of the attack. 
To run the probabilistic attack, you can use this example, setting the `--attack` argument to prob.
In addition, compared to the TrojanZoo framework, there are a few additional arguments used specifically for ProbTrojan.
These include:

`--extra_mark`: Used to specify additional triggers.  
In Trojanzoo, there are some command line arguments to define the trigger. These include:
- `mark_path`
- `mark_width` 
- `height_offset`
- `width_offset`

ProbTrojan uses more than one trigger. One of them is specified with the above arguments.
The other ones must be specified with another argument called `--extra_mark`.
It can be used several times,
and is a dictionary argument that can accept as keys all the four above mentioned arguments.

example using three triggers:

```
--mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2
--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10"
--extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20"
```

`--probs`: a list of desired per-trigger ASRs.  
The list must have as many elements as the triggers.
A reasonable choice is to set all of them to the reciprocal of the number of triggers.  
example: `--probs 0.25 0.25 0.25 0.25`

`--losses`: a list of loss function names used in training the network as an attacker.  
For a list of losses refer to [losses.py](./trojanvision/attacks/backdoor/prob/losses.py).
We used `loss1 loss2_11 loss3_11` in the paper.

For an example of the ProbTrojan attack, see [prob_test.py](./prob_test.py).
