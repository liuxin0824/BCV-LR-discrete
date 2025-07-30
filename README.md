# BCV-LR
### R1

We appreciate your careful reading and professional comments! In response to your feedback, we have supplemented the missing important works and further conducted several supplementary experiments, which greatly improves the quality of the paper. We hope these revisions will meet your expectations!

___
>**W1:I am not sure whether the offline stage is individually conducted on each task or jointly conducted on a multi-task dataset. The practical significance of video behavior cloning in a single-task setting is limited.**

In the submitted version, all experiments are conducted under single-task settings. This is mainly due to two reasons: first, whether it is possible to balance video imitation performance and sampling efficiency in single-task settings remains an open problem; second, the recent advanced pure ILV (without reward) works mainly achieve evaluation under single-task settings. We agree with the reviewer that achieving video cloning under multi-task settings are more meaningful! As you suggested, we have supplemented multi-task experiments in a fashion akin to FICC, as shown in answer to Q3.
___
>**W2:The LAPO in the baseline is designed for the pre-training on multi-task data, which may affect comparisons.**

LAPO has indeed inspired many works on multi-task pre-training. However, according to the final published version of LAPO and its open-source code, it uses a 'full' distribution for each task in Procgen to ensure intra-task diversity (e.g., the 'maze' task includes mazes of different sizes, colors, and layouts) while does not perform cross-task 'multi-task learning' (i.e., it achieves pre-training on 'maze' videos and then performs online learning in the 'maze' task). The Procgen experiments of BCV-LR have strictly followed LAPO's setup in terms of environment configuration and we use the same video dataset as that of LAPO, thus ensuring a fair comparison. Please feel free to let us know if you have any further questions!
___
>**W3:Additionally, incorporating extra ablation studies and baselines could further solidify this work.**

According to your constructive reviews, we have further supplemented several additional experiments and improve our work!
___
>**Q1:Does the phrase "without the need to access any other supervision" include the supervision of actions during the online phase? The supervision on the actions directly injects the environment's inverse dynamics knowledge and potentially adjusts the distribution of latent actions through alignment with ground-truth actions. If the phrase refers to merely using the latent reward information implicit in expert videos without online expert rewards for video behavior cloning, JEPT[1] may be an uncited but related work. Similarly, without using an explicit reward signal, JEPT can achieve generalization of one-shot visual imitation in some tasks with a mixture dataset.**

Thank you for your careful reading! BCV-LR does utilize information from environmental actions and our original intention was to emphasize that it does not rely on 'expert' action labels. We have replaced "without the need to access any other supervision" with "without the need to access any other expert supervision" to eliminate any ambiguity. Additionally, JPET[1] utilizes a mixed dataset containing video data for reward-free policy learning, which is relevant to our work. We have cited JPET and discussed it in the related work section (Please note that we are temporarily unable to update the paper in the system.).
___
>**Q2:The design of BCV-LR is very similar to FICC[2] (mainly in the offline part), while FICC is not properly referenced. FICC should be a comparable baseline in the discrete setting.**

Thanks for your careful reading and we indeed missed this important related work[2]. We have However, due to the following reasons, we have not yet included FICC in the comparative experiments on ProcGen:

1. FICC only provides code for the pre-training part. It does not offer code for fine-tuning the pre-trained model through interaction in the environment and combining it with MCTS-based RL, making it difficult for us to reproduce this work in a short period of time.
2. FICC has not been tested on the ProcGen benchmark. Moreover, whether its MCTS-based backbone EfficientZero (EZ)[3] is applicable to procedurally generated environments remains an open question, which leaves us without reference information such as hyperparameter settings and thus unable to ensure the reasonability of reproductions.
3. We have already compared BCV-LR against 6 advanced and popular baselines, including the state-of-the-art RL algorithms on ProcGen. Additionally, none of our baselines have been directly compared with the FICC algorithm. Therefore, we believe the current comparisons on Procgen are sufficient, which is also recognized by reviewer HHys.
4. The backbone of FICC, EZ, is based on MCTS, requiring a significant amount of training time. According to EZ[3], it requires 28 GPU hours for 100k steps training, while our BCV-LR only requires about 1 GPU hour. This high consumption makes many subsequent works [4,5] to avoid direct comparisons with EZ even in experiments on Atari. This also makes it impossible for us to obtain its results in a short period of time.

If you have any questions, please let us know, thanks!
>**Q3:I am curious whether the effectiveness of direct behavior cloning in the online phase is related to the single-task setting or the quality of expert data. The offline phase of BCV-LR is quite similar to LAPO and FICC, while both LAPO and FICC avoid direct behavior cloning in the online phase and choose to use online rewards for policy adjustment in the multitask setting. It would be better if a direct online BC variant of LAPO were involved in the comparison, which means the online reward will also be excluded in this setting, and may further illustrate this aspect.**

As we illustrated in answer to W2, LAPO also focuses on single-task setting. FICC indeed provides multi-task pre-training experiments while also conducts single-task experiments as their main results. In addition, the video dataset we use in Procgen is consistent with that of LAPO, which means the quality of expert data is the same for all video-based methods. As per your request, we have conducted extra experiments where we replaced the reinforcement learning loss of LAPO with the BC loss and compared it with our BCV-LR. The results in Table I demonstrate that LAPO-BC enables effective policy learning in some tasks while our BCV-LR still exhibits performance advantages. In addition, we also show that our BCV-LR can generalize to multi-task pre-training, which is detailed in answers to Q4.
**Table I**
| task  | BCV-LR |  LAPO-BC |  PPO | / |video|
| - | - | - | - | - | - | 
| fruitbot |  **27.5 ± 1.5** |  6.2 ± 1.9  | -1.9 ± 1.0  | | 29.9 |
| heist  | **9.3 ± 0.1**  | 9.2 ± 0.3  | 3.7 ± 0.2 || 9.7 |
| bossfight  | **10.3 ± 0.3**  | 0.0 ± 0.0  | 0.1 ± 0.1 || 11.6 |
| chaser  | **3.1 ± 0.5**  | 0.6 ± 0.0  | 0.4 ± 0.2 || 10.0|

>**Q4:The paper lacks experiments in a multi-task setting or a discussion on the relationship between the expert video scale and the performance in the single-task setting. Can BCV-LR generalize to new tasks through multi-task data? Or, for new tasks lacking sufficient data, how much offline data is required for BCV-LR to fulfill behavior cloning? Otherwise, the prerequisite of obtaining sufficient data often implies having a well-performing in-domain policy on the task, reducing the significance of behavior cloning from expert videos.**

First, we conducted additional experiments in Procgen to test the multi-task pre-training ability of BCV-LR, which follows the settings of FICC mutli-task experiments. Concretely, we pre-trains one model on mixed videos of 'bigfish', 'maze', and 'starpilot', and then finetunned the pre-trained models in these seen tasks seperately. Different from FICC, we also employ two unseen tasks into evaluations. The results in Table II show that BCV-LR enables effective policy imitation on all tasks. It achieves robust multi-task pre-training, where the pre-trained knowledge can be shared across both seen and unseen domains.

**Table II**
|||BCV-LR-MT(share pre-training)|/|PPO|BCV-LR|
|-|-|-|-|-|-| 
|seen|bigfish|32.2 ± 1.0||0.9 ± 0.1|35.9|
||maze|9.6 ± 0.1||5.0 ± 0.7|9.9|
||starpilot|44.3 ± 1.9||2.6 ± 0.9|54.8|
|unseen|bossfight|5.5 ± 0.3||0.1 ± 0.1|10.3|
||dodgeball|9.5 ± 0.3||1.1 ± 0.2|12.4|

Then, we provide the video data efficiency experiments. We provide BCV-LR with 5k, 20k, 50k, and 100k expert video transitions. The results shown i

**Table III**
|Video data of BCV-LR|5k|20k|50k|100k|/|video|
|-|-|-|-|-|-|-| 
|reacher_hard|0 ± 0|384 ± 153|799 ± 34|**900 ± 31**||967|
|finger_spin|596 ± 17|901 ± 33|905 ± 70|**942 ± 48**||981|


___
[1]Learning Video-Conditioned Policy on Unlabelled Data with Joint Embedding Predictive Transformer. ICLR 25
[2]Become a Proficient Player with Limited Data through Watching Pure Videos. ICLR 23
[3]Mastering Atari Games with Limited Data. Neurips 21
[4]Value-Consistent Representation Learning for Data-Efficient Reinforcement Learning. AAAI 23
[5]Mask-based Latent Reconstruction for Reinforcement Learning. Neurips 22
[6]Learning to act without actions. ICLR 24



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

