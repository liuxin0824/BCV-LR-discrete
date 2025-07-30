# BCV-LR


### R3

>**Q1：Stage 3. (as per the summary above), in which the latent action model is finetuned and the action decoder is trained, requires real actions to be observed via environment interaction. It is unclear what is the impact of the schedule of environment interactions, how exactly the phases are optimally alternated so as to sample iteratively from the policy learned online via the concurrent step 4. Does this matter?**

The BCV-LR online stage contains three steps. First, we (1) allow the agent to interact with the environment for a fixed number of steps using its policy and enrich the experience buffer. Immediately after that, (2) we perform finetuning of the latent action and training of the action decoder on the experience buffer. Then, (3) using expert videos, we train the latent policy to imitate the finetuned latent action. After this, we will return to step (1) and complete the cyclic online policy learning. This alternation is intuitive, that is, first collect new data of higher quality, then use the new data to fine-tune the latent action, and finally let the policy learn the fine-tuned latent action to achieve performance improvement. You can refer to the Pseudo Code 

In previous experiments, we did not adjust the number of steps for each interaction, but kept it fixed at a relatively large value from start to finish (for example, we fixed the step number of step (1) as 1000) and ultimately achieved satisfactory results. To answer your question, we further attempted to use a smaller number of steps (updating every two interactions) and correspondingly reduced the number of updates for latent actions (once) and latent policies (twice) after each interaction, making BCV-LR in a fashion akin to off-policy RL. The results in Table I show that it can still achieve effective learning when the step number is reduced, which demonstrates the robustness of our method.
| tasks  | BCV-LR (1000)  | BCV-LR (2) | BCO  |UPESV |TACO  | DrQv2| / |video|
| - | - | - | - | - | - |  - | - | - |
| reacher_hard | 900 ± 31   | 158  | 336  | 18     | 310  | 232 | / | 698|
| finger_spin  | 942 ± 48   | 0.20  | 0.31  | 0.03     | 0.45  | 0.34 | /| 1.00|



### R4
___
>**Q1：How general is the learned encoder? Is the shift image + contrastive loss learn useful features for non-video games, say real world robotics tasks? Or will this only work for relatively simplistic videos/environments?**

We apply this ‘shift image + contrastive loss’ (Eq.1, Line 185, Mainpaper) for Procgen video games because it has been proven effective in these kinds of tasks. As shown in Sec.3.1.1 (Line 174, Mainpaper), BCV-LR is easily compatible with any action-free self-supervised tasks and it can adapt to different types of domains by choosing appropriate self-supervised objectives. This motivates us to choose another prototype-based temporarl loss (Eq.10, Line 549, Appendix) for Deepmind control tasks in our experiments, where the temporarl understanding is crucial for agents and this temporarl loss has been proven better than contrastive loss in previous works. To this end, BCV-LR can involve more advanced self-supervised objective (e.g., ViT-based masked reconstruction) for more challenging tasks (e.g., real-world tasks) if necessary. In addition, BCV-LR can also be combined with an off-the-shelf well-trained encoder, which makes its potential not limited to only video game environments.
___

>**Q2:How well does the mapping from latent action to action work in more complex environments, e.g., robot controls rather than just video game controls?**

In addition to the video games, we also demonstrate the advantages of BCV-LR on DMControl benchmark which consists of several continuous robotic control tasks. We summarize the average results below. More details are provided in Sec.4.3 (mainpaper) and Sec.C.1 (Appendix).

| DMControl-8-tasks  | BCV-LR(ours)   | LAIFO   | BCO  |UPESV |TACO  | DrQv2| / |video|
| - | - | - | - | - | - |  - | - | - |
| Mean Score | **604**   | 158  | 336  | 18     | 310  | 232 | / | 698|
| Video-norm Mean Score  | **0.78**   | 0.20  | 0.31  | 0.03     | 0.45  | 0.34 | /| 1.00|


In addition, we further conducted additional experiments in Metaworld manipulation benchmark. Only 50k environmental steps are allowed for each Metaworld task, with remaining settings similar to that of DMControl. Results are shown below. In this interaction-limited situation, BCV-LR can still derive effective manipulation skills from expert videos without accessing expert actions and rewards, which demonstrates its wider range of applications and potential for generalizing to real-world manipulation tasks.

| Metaworld  | BCV-LR(ours) | BCO   | DrQv2| / |video|
| - | - | - | - | - | - | 
| Faucet-open| **0.82 ± 0.20**   | 0.13 ± 0.19  | 0 ± 0 | /  | 1.00 |
| Reach| **0.63 ± 0.25**   | 0.03 ± 0.05  |  | /  | 1.00 |
| Drawer-open| **0.92 ± 0.12**   | 0.13 ± 0.09  |0 ± 0  | /  | 1.00 |
| Faucet-close| **0.98 ± 0.04**   | 0 ± 0 | 0.70 ± 0.30 | /  | 1.00 |
| Mean SR| **0.84**   | 0.07  |  | /  | 1.00 |

___

