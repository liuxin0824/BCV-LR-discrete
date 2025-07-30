# BCV-LR
### R1

We appreciate your careful reading and professional comments! In response to your feedback, we have supplemented the missing important works and further conducted several supplementary experiments, which greatly improves the quality of the paper. We hope these revisions will meet your expectations!

___
>**W1:I am not sure whether the offline stage is individually conducted on each task or jointly conducted on a multi-task dataset. The practical significance of video behavior cloning in a single-task setting is limited.**

In the submitted version, all experiments are conducted under single-task settings. This is mainly due to two reasons: first, whether it is possible to balance video imitation performance and sampling efficiency in single-task settings remains an open problem; second, the recent advanced pure ILV (without reward) works mainly achieve evaluation under single-task settings. We agree with the reviewer that achieving video cloning under multi-task settings are more meaningful! As you suggested, we have supplemented multi-task experiments in a fashion akin to FICC, as shown in answer to Q3.
___
>**W2:The LAPO in the baseline is designed for the pre-training on multi-task data, which may affect comparisons.**

LAPO has indeed inspired many works on multi-task pre-training. However, according to the final published version of LAPO and its open-source code, it uses a 'full' distribution for each task in Procgen to ensure intra-task diversity (e.g., the 'maze' task includes mazes of different sizes, colors, and layouts) while does not perform cross-task 'multi-task learning' (i.e., it achieves pre-training on 'maze' videos and then performs online learning in the 'maze' task). The Procgen experiments of BCV-LR have strictly followed LAPO's setup in terms of environment configuration, thus ensuring a fair comparison. Please feel free to let us know if you have any further questions!
___
>**W3:Additionally, incorporating extra ablation studies and baselines could further solidify this work.**

According to your constructive reviews, we have further supplemented several additional experiments and improve our work!
___
>**Q1:Does the phrase "without the need to access any other supervision" include the supervision of actions during the online phase? The supervision on the actions directly injects the environment's inverse dynamics knowledge and potentially adjusts the distribution of latent actions through alignment with ground-truth actions. If the phrase refers to merely using the latent reward information implicit in expert videos without online expert rewards for video behavior cloning, JEPT[1] may be an uncited but related work. Similarly, without using an explicit reward signal, JEPT can achieve generalization of one-shot visual imitation in some tasks with a mixture dataset.**

As reviewer says, BCV-LR utilizes the knowledge 





### R3
We are deeply grateful to the reviewer for your meticulous reading and positive recognition of our work! We have further enhanced our paper in light of your feedback and answered your questions 
___
>**Q1&Weakness2：Stage 3. (as per the summary above), in which the latent action model is finetuned and the action decoder is trained, requires real actions to be observed via environment interaction. It is unclear what is the impact of the schedule of environment interactions, how exactly the phases are optimally alternated so as to sample iteratively from the policy learned online via the concurrent step 4. Does this matter?**

The BCV-LR online stage contains three parts. First, we a) allow the agent to interact with the environment for a fixed number of steps using its policy and enrich the experience buffer. Immediately after that, b) we perform finetuning of the latent action and training of the action decoder on the experience buffer. Then, c) using expert videos, we train the latent policy to imitate the finetuned latent action. After this, we will return to part a) and complete the cyclic online policy learning. This alternation is intuitive, that is, first collect new data of higher quality, then use the new data to fine-tune the latent action, and finally let the policy learn the fine-tuned latent action to achieve performance improvement. You can refer to Sec.A.1 (Appendix) for Pseudo Code.

In previous experiments, we did not adjust the number of steps for each interaction, but kept it fixed at a relatively large value from start to finish (for example, we fixed the step number of step (1) as 1000) and ultimately achieved satisfactory results. To answer your question, we conduct additional experiments, where use a smaller interation number (1000->2) and correspondingly reduced the number of update times for latent actions (100->1) and latent policies (1000->2) after each interaction, making BCV-LR perform in a fashion akin to off-policy RL. The results in Table I show that it can still achieve effective learning when the step number is reduced, which demonstrates that BCV-LR is robust to the schedule of environment interactions.
**Table I**
| task  | BCV-LR(1000->2) |  BCV-LR |   DrQv2| / |video|
| - | - | - | - | - | - | 
| reacher_hard |  875 ± 65 |  **900 ± 31**  | 92 ± 98  | | 967 |
| finger_spin  | **956 ± 20**  | 942 ± 48  | 374 ± 264 || 981|
___
>**Q2&Weakness4: How does the method compare to performing the steps in a fashion akin to LAPA? (No online learning, only imitation learning alignment on ground truth expert action?**

To answer your questions, we further conduct additional experiments, where we let BCV-LR performs latent action finetuning and policy imitation with a few action-labeled expert transitions. Concretely, we maintain the original pre-training stage, while use 10k offline expert transitions to achieve offline imitation learning alignment. The results in Table (RL denotes DrQv2 for DMControl while PPO for procgen) demonstrate that BCV-LR can also well achieve offline imitation learning alignment. Of course, we would like to say that BCV-LR is designed for the ILV (imitation learning from videos), a much harder variant of the classical ILO (imitation learning from observation only) problem, where the employment of unsupervised online policy training adheres to the norms of this field.

**Table II**
| task  | BCV-LR-offline |  BCV-LR |  RL | / |video|
| - | - | - | - | - | - | 
| reacher_hard |  **938 ± 44** |  900 ± 31  | 92 ± 98  | | 967 |
| finger_spin  | **978 ± 7**  | 942 ± 48  | 374 ± 264 || 981|
| fruitbot  | **27.7 ± 0.4**  | 27.5 ± 1.5  | -1.9 ± 1.0 || 29.9|
___

>**Weakness3: The method as requested does require a source of ground truth actions from the videos, it is thus not strictly suitable for solving pressing hard problems for real world robotics, e.g. learning from real human videos and then transferring the policies to robots. This is also related to the benchmarks being limited to classic "toy RL" simulation environments.**

As the reviewer says,  our method has not yet been extended to more challenging real-world tasks, nor has it sufficiently explored knowledge transfer. We would like to provide some additional results, which may demonstrate the potential of our method in more task settings and application scenarios. First, we 

In addition, we further conducted additional experiments in Metaworld manipulation benchmark. Only 50k environmental steps are allowed for each Metaworld task, with remaining settings similar to that of DMControl. Results are shown below. In this interaction-limited situation, BCV-LR can still derive effective manipulation skills from expert videos without accessing expert actions and rewards, which demonstrates its wider range of applications and potential for generalizing to real-world manipulation tasks.

| Metaworld  | BCV-LR | BCO   | DrQv2| / |video|
| - | - | - | - | - | - | 
| Faucet-open| **0.82 ± 0.20**   | 0.13 ± 0.19  | 0.00 ± 0.00 |  | 1.00 |
| Reach| **0.63 ± 0.25**   | 0.03 ± 0.05  | 0.13 ± 0.12 |  | 1.00 |
| Drawer-open| **0.92 ± 0.12**   | 0.13 ± 0.09  | 0.00 ± 0.00  |  | 1.00 |
| Faucet-close| **0.98 ± 0.04**   | 0 ± 0 | 0.50 ± 0.28 |  | 1.00 |
| Mean SR| **0.84**   | 0.07  | 0.16 |   | 1.00 |







### R4
We greatly appreciate the reviewer's careful reading and recognition of our work! We have carefully reviewed your comments, addressed your questions, and further improved our paper based on your feedback!
___
>**Q1：How general is the learned encoder? Is the shift image + contrastive loss learn useful features for non-video games, say real world robotics tasks? Or will this only work for relatively simplistic videos/environments?**

We apply this ‘shift image + contrastive loss’ (Eq.1, Line 185, Mainpaper) for Procgen video games because it has been proven effective in these kinds of tasks. As shown in Sec.3.1.1 (Line 174, Mainpaper), BCV-LR is easily compatible with any action-free self-supervised tasks and it can adapt to different types of domains by choosing appropriate self-supervised objectives. This motivates us to choose another prototype-based temporarl loss (Eq.10, Line 549, Appendix) for Deepmind control tasks in our experiments, where the temporarl understanding is crucial for agents and this temporarl loss has been proven better than contrastive loss in previous works. To this end, BCV-LR can involve more advanced self-supervised objective (e.g., ViT-based masked reconstruction) for more challenging tasks (e.g., real-world tasks) if necessary. In addition, BCV-LR can also be combined with an off-the-shelf well-trained encoder, which makes its potential not limited to only video game environments.
___

>**Q2:How well does the mapping from latent action to action work in more complex environments, e.g., robot controls rather than just video game controls?**

In addition to the video games, we also demonstrate the advantages of BCV-LR on DMControl benchmark which consists of several continuous robotic control tasks. We summarize the average results below. More details are provided in Sec.4.3 (mainpaper) and Sec.C.1 (Appendix).

| DMControl-8-tasks  | BCV-LR  | LAIFO   | BCO  |UPESV |TACO  | DrQv2| / |video|
| - | - | - | - | - | - |  - | - | - |
| Mean Score | **604**   | 158  | 336  | 18     | 310  | 232 | | 698|
| Video-norm Mean Score  | **0.78**   | 0.20  | 0.31  | 0.03     | 0.45  | 0.34 | | 1.00|


In addition, we further conducted additional experiments in Metaworld manipulation benchmark. Only 50k environmental steps are allowed for each Metaworld task, with remaining settings similar to that of DMControl. Results are shown below. In this interaction-limited situation, BCV-LR can still derive effective manipulation skills from expert videos without accessing expert actions and rewards, which demonstrates its wider range of applications and potential for generalizing to real-world manipulation tasks.

| Metaworld  | BCV-LR | BCO   | DrQv2| / |video|
| - | - | - | - | - | - | 
| Faucet-open| **0.82 ± 0.20**   | 0.13 ± 0.19  | 0.00 ± 0.00 |  | 1.00 |
| Reach| **0.63 ± 0.25**   | 0.03 ± 0.05  | 0.13 ± 0.12 |  | 1.00 |
| Drawer-open| **0.92 ± 0.12**   | 0.13 ± 0.09  | 0.00 ± 0.00  |  | 1.00 |
| Faucet-close| **0.98 ± 0.04**   | 0 ± 0 | 0.50 ± 0.28 |  | 1.00 |
| Mean SR| **0.84**   | 0.07  | 0.16 |   | 1.00 |

___

