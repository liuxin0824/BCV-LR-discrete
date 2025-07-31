# BCV-LR rebuttal
## R1

We appreciate your careful reading, professional comments, and affirmation of our paper! In response to your feedback, we have supplemented the missing related works and further conduct several supplementary experiments, which greatly improves the quality of the paper. We hope these revisions will meet your expectations!

___
>**W1:I am not sure whether the offline stage is individually conducted on each task or jointly conducted on a multi-task dataset. The practical significance of video behavior cloning in a single-task setting is limited.**

In the submitted version, all experiments are conducted under single-task settings. This is mainly due to two reasons: first, whether it is possible to balance video imitation performance and sampling efficiency in single-task settings remains an open problem before our paper; second, the recent advanced pure ILV (without reward) works mainly achieve evaluation under single-task settings. We agree with the reviewer that achieving video cloning under multi-task settings are more meaningful and supplement multi-task experiments as you suggested, as shown in answer to Q4.
___
>**W2:The LAPO in the baseline is designed for the pre-training on multi-task data, which may affect comparisons.**

LAPO has indeed inspired many works on multi-task pre-training. However, according to the final published version of LAPO and its open-source code, it uses a 'full' distribution for each task in Procgen to ensure intra-task diversity (e.g., the 'maze' task includes mazes of different sizes, colors, and layouts) while does not perform cross-task 'multi-task learning' (i.e., it achieves pre-training on 'maze' videos and then performs online learning in the 'maze' task). The Procgen experiments of BCV-LR have strictly followed LAPO's setup in terms of environment configuration and we use the same video dataset as that of LAPO, thus ensuring a fair comparison. Please feel free to let us know if you have any further questions!

___
Due to space constraints, we can't list your concrete questions. Please refer to your review section. We apologize for the inconvenience!
>**answer to Q1: We supplement the reference and discussion of JPET.**

Thank you for your careful reading! BCV-LR does utilize information from environmental actions and our original intention was to emphasize that it does not rely on 'expert' action labels. We have replaced "without the need to access any other supervision" with "without the need to access any other expert supervision" to eliminate any ambiguity. Additionally, JPET[1] utilizes a mixed dataset containing video data for reward-free policy learning, which is very relevant to our work and we missed it. We now add the citation and discussion of JPET in the related work (Please note that we are temporarily unable to update the paper now).
___
>**answer to Q2: We add the reference and discussion of FICC and explain why it has not been included in the comparison for the time being.**

Thanks for your careful reading, and we now have supplemented FICC [2] in the related work. That said, due to the following reasons, we have not yet included FICC in the experiments on ProcGen:

1. FICC provides code for its pre-training part while doesn't offer code for fine-tuning the pre-trained model and combining it with its MCTS-based backbone online. This makes it difficult for us to reproduce this work within the short timeframe of the rebuttal period.
2. FICC has not been evaluated on the Procgen. Moreover, whether its MCTS-based backbone EfficientZero (EZ)[3] is applicable to procedurally generated environments remains an open question. This lack of information—such as hyperparameter settings and EZ framework in Procgen—leaves us unable to ensure the validity of our reproductions.
3. We have already compared BCV-LR against 6 advanced and popular baselines on Procgen, including the state-of-the-art RL and ILV algorithms. Additionally, to our knowledge, none of these methods have been directly compared with the FICC. Thus, we feel that the current set of comparisons on Procgen is adequate, a view that also aligns with the feedback from Reviewer HHys and TLPv.
4. The backbone of FICC, EZ, is built on MCTS and thus requires considerable training time. As noted in EZ [3], it take 28 GPU hours for 100k steps in discrete control, whereas BCV-LR online stage requires only around 1 GPU hour on both discrete and continuous domains. This relatively high computational cost has led many subsequent works [4,5] to refrain from direct comparisons with EZ, even in Atari experiments. In addition, this consumption also means it's hard to obtain its results within the limited timeframe available.

If you have any questions, please feel free to let us know!
___
>**answer to Q3: We explain that the single-task setting and training data don't cause unfairness. We conduct extra experiments on LAPO's BC variant.**

As we explain in our answer to W2, LAPO also focuses on single-task settings. FICC does include multi-task pre-training experiments, but its main results are from single-task experiments.​
What’s more, the Procgen video we use is the same as LAPO’s original data, as stated in Sec.4.2 (line286 mainpaper). This means all video-based methods have expert data of the same quality.​
As you asked, we conduct extra experiments: we replaced LAPO’s online RL loss with BC loss, then compared it with our BCV-LR. The results below show that LAPO-BC works well for policy learning in some tasks, but our BCV-LR still performs better.​
We also show that BCV-LR can be used for multi-task pre-training in a fashion akin to FICC. Please refer to answer to Q4.

|Task|BCV-LR|LAPO-BC|PPO|/|video|
|-|-|-|-|-|-| 
|fruitbot|**27.5 ± 1.5**|6.2 ± 1.9|-1.9 ± 1.0||29.9|
|heist|**9.3 ± 0.1**|9.2 ± 0.3|3.7 ± 0.2||9.7|
|bossfight|**10.3 ± 0.3**|0.0 ± 0.0|0.1 ± 0.1||11.6|
|chaser|**3.1 ± 0.5**|0.6 ± 0.0|0.4 ± 0.2||10.0|
___
>**answer to Q4: We conduct extra experiments on multi-task ability of BCV-LR and provide video data efficiency analysis.**

According to your suggestions, we conducted additional multi-task experiments for BCV-LR in Procgen, which follows the settings of FICC mutli-task pre-training experiments. Concretely, we pre-trains one BCV-LR model on mixed videos of 'bigfish', 'maze', and 'starpilot', and then finetunned it in these seen tasks seperately. Different from FICC, we further employ two unseen tasks into evaluations. The results below show that BCV-LR enables effective policy imitation on all tasks. It achieves robust multi-task pre-training, where the pre-trained knowledge can be shared across both seen and unseen domains.

||Task|BCV-LR(multi-task)|PPO(single-task)|/|BCV-LR(single-task)|
|-|-|-|-|-|-| 
|seen|bigfish|**32.2 ± 1.0**|0.9 ± 0.1||35.9|
||maze|**9.6 ± 0.1**|5.0 ± 0.7||9.9|
||starpilot|**44.3 ± 1.9**|2.6 ± 0.9||54.8|
|unseen|bossfight|**5.5 ± 0.3**|0.1 ± 0.1||10.3|
||dodgeball|**9.5 ± 0.3**|1.1 ± 0.2||12.4|

Then, we provide the video data efficiency experimental results. We test BCV-LR with 5k, 20k, 50k, and 100k (default in main experiments) video transitions. Results demonstrate that BCV-LR enables effective policy learning with only 20k transitions. 50k transitions can support near-expert policy performance. Refer to Appendix C.5 for curves and more details.

|Video data of BCV-LR|5k|20k|50k|100k|/|video|
|-|-|-|-|-|-|-| 
|reacher_hard|0 ± 0|384 ± 153|799 ± 34|**900 ± 31**||967|
|finger_spin|596 ± 17|901 ± 33|905 ± 70|**942 ± 48**||981|
___
>**answer to Q5: We conduct extra ablation experiments.**

As per your suggestions, we first conducted additional experiments, where we finetune the self-supervised encoder with $L_{la}$ and $L_{ft}$ repectively. The results in Table IV demonstrate that whether finetuning self-supervised visual representation doesn't yield apprent effct on policy performance (curves are also similar). This phenomenon has also been observed in self-supervised RL[7], leading some works to fine-tune self-supervised representations while others opt to freeze them.

|Task|Finetuning with $L_{la}$|Finetuning with $L_{ft}$|No visual finetuning|
|-|-|-|-|
|reacher_hard|876 ± 15|**906 ± 65**|900 ± 31|
|finger_spin|937 ± 26|920 ± 57|**942 ± 48**|

Then, we provide the experiments to show the impact on performance of not updating the latent action predictor with $L_{ft}$ in the online phase. Partial results are presented in Table V (denoted as BCV-LR w/o ft), demonstrating that utilizing the environmental actions to finetune pre-trained latent actions is a crucial step in BCV-LR, especially in DMControl tasks. Please refer to Sec.4.4 for more ablation results.

||Task|BCV-LR|BCV-LR w/o ft|
|-|-|-|-|
|DMControl|point_mass_easy|**800 ± 25**|22 ± 2|
||jaco_reach_bottom_left|**123 ± 39**|15 ± 10|
|Procgen|starpilot|**54.8 ± 1.4**|29.2 ± 10.6|
||fruitbot|**27.5 ± 1.5**|24.2 ± 2.0|

___
In addition, we further conducted additional experiments in Metaworld. Only 50k steps are allowed for each task, with remaining settings similar to that of DMControl. Results demonstrates BCV-LR's wider range of applications and potential for generalizing to real-world manipulation tasks.

|Metaworld-50k|BCV-LR|BCO|DrQv2|/|video|
|-|-|-|-|-|-| 
|Faucet-open|**0.82 ± 0.20**|0.13 ± 0.19|0.00 ± 0.00||1|
|Reach|**0.63 ± 0.25**|0.03 ± 0.05|0.13 ± 0.12||1|
|Drawer-open|**0.92 ± 0.12**|0.13 ± 0.09|0.00 ± 0.00||1|
|Faucet-close|**0.98 ± 0.04**|0.00 ± 0.00|0.50 ± 0.28||1|
|Mean SR|**0.84**|0.07|0.16||1|

___
[1]Learning Video-Conditioned Policy on Unlabelled Data with Joint Embedding Predictive Transformer.ICLR25
[2]Become a Proficient Player with Limited Data through Watching Pure Videos.ICLR23
[3]Mastering Atari Games with Limited Data.Neurips21
[4]Value-Consistent Representation Learning for Data-Efficient Reinforcement Learning.AAAI23
[5]Mask-based Latent Reconstruction for Reinforcement Learning.Neurips22
[6]Learning to act without actions.ICLR24
[7]Decoupling representation learning from reinforcement learning.ICML21

















## R2
We greatly appreciate the reviewer’s careful reading, detailed feedback, and recommendation of our paper! We have carefully reviewed your comments and further refined our paper based on your constructive reviews!

___
>**W:The scope of the experiments are relatively limited. All experiments are conducted in simulation/"toy" environments (Procgen and DMControl). While these are standard benchmarks, it is unclear how well BCV-LR generalizes to internet-scale human videos and real-world robotics applications.**

We understand your concern! Since it is highly challenging to balance video imitation performance and efficiency without access to expert actions or expert rewards, our experiments have primarily been conducted in relatively standard visual RL environments to answer this open question, and we have not extended our work to real-world tasks or considered internet-scale pre-training, which are of greater significance. Based on your feedback, we further conduct some extra experiments in Metaworld manipulation benchmark, which may demonstrate a wider application of BCV-LR.

For each Metaworld task, only 50k environmental steps are allowed, with remaining settings similar to that of DMControl. Results are shown in Table III. In this interaction-limited situation, BCV-LR can still derive effective manipulation skills from expert videos without accessing expert actions and rewards, which demonstrates its wider range of applications and potential for generalizing to real-world manipulation tasks. Furthermore, we would like to add a discussion on real-world tasks and internet-scale knowledge transferring in the 'Limitation and Future Work' sections to make our paper more comprehensive.

**Table I**
|Metaworld-50k|BCV-LR|BCO|DrQv2|/|video|
|-|-|-|-|-|-| 
|Faucet-open|**0.82 ± 0.20**|0.13 ± 0.19|0.00 ± 0.00||1|
|Reach|**0.63 ± 0.25**|0.03 ± 0.05|0.13 ± 0.12||1|
|Drawer-open|**0.92 ± 0.12**|0.13 ± 0.09|0.00 ± 0.00||1|
|Faucet-close|**0.98 ± 0.04**|0.00 ± 0.00|0.50 ± 0.28 ||1|
|Mean SR|**0.84**|0.07|0.16||1|
___
>**Q1:For continuous control tasks, they are relatively easy with low-dimensional action spaces. I'm curious how would the proposed methods work for environments with high-dimensional action spaces. (For example, take humanoid & dog from Deepmind Control Suite)**

Following your suggestion, we attempt experiments on the 'dog' and 'humanoid' domains. We find that these two domains are highly challenging even for advanced RL methods that use expert rewards. Specifically, we first try training agents to collect expert data via RL, but observe that both TACO and DrQ fail within 5M steps (obtaining scores lower than 5). Furthermore, we run DrQv2 on the 'humanoid_walk' task for 20M steps across 4 seeds, with only 1 run successfully learning a policy. We use this 20M-step policy to collect data, and then train BCV-LR and all baselines under the 100k-step setting—all of which failed. (Table II) This indicates that complex dynamics and action spaces remain significant challenges for current policy learning algorithms, even when expert rewards are provided. Meanwhile, BCV-LR, which is designed to balance sample efficiency under more challenging settings (without access to rewards or expert actions), is not yet able to handle such tasks effectively. We would like to add the above discussion to the 'Limitations and Future Work' sections to make our paper more comprehensive.
**Table II**
|  | BCV-LR  | LAIFO   | BCO  |UPESV |TACO  | DrQv2| / |video|
| - | - | - | - | - | - |  - | - | - |
| humanoid_walk-100k |1.3 ± 0.1  | 1.1 ± 0.0 | 1.1 ± 0.1 | 1.0 ± 0.1 | 2.0 ± 0.2 | 1.9 ± 0.2| |529|

___
>**Q2:The author mention that during self-supervised representation learning phase, it also learns some temporal dynamics by aligning representation of $o_t$ with $o_{t+1}$, but this seems not appearing in Equation 1. So what's the overall objective for this stage. Is it Equation 1 (contrastive loss with image reconstruction ) plus temporal loss?**

As shown in Sec.3.1.1 (line 174, main paper), BCV-LR is designed to be easily compatible with any action-free self-supervised tasks (it completely decouples visual learning from subsequent training), and it can adapt to different types of tasks by choosing appropriate self-supervised objectives. To this end, we apply different latent feature training losses for Procgen and DMControl. Concretely, we employ the 'contrastive loss with image reconstruction' (Eq.1, line 185, main paper) for Procgen video games, because it has been proven effective in these kinds of tasks [1,2]. For DMControl environments where the temporal understanding is crucial, we choose another prototype-based temporal association loss (Eq.10, line 549, Appendix) because invoving temporal information into self-supervised objectives have been proven necessary in DMControl [3]. In summary, BCV-LR can involve more advanced self-supervised objectives for more challenging tasks if necessary, which makes its potential not limited to specific environments.

___
>**Q3:The reconstruction loss of feature learning seems suboptimal especially for environments with distracting backgrounds. The authors mention that they get better performance with this loss added. But is it because the visual observations of the environments it tests on are too simple?**

We agree with your opinion on reconstruction loss and some previous works [6] have demonstrated the limitation of reconstruction when faced with visual distraction. As we explain in our answer to Q2, BCV-LR employs the reconstruction loss in Procgen because of its effectiveness in video games [1], but it is not limited to this reconstruction loss. BCV-LR is designed to be easily compatible with any action-free self-supervised tasks (it completely decouples visual learning from subsequent training), and it can adapt to different types of tasks by choosing appropriate self-supervised objectives. To this end, BCV-LR can involve more advanced self-supervised objectives (e.g., ViT-based masked reconstruction [4,5] or distraction-robust prototypical representation [6]) for more challenging tasks if necessary. In addition, BCV-LR can also be combined with an off-the-shelf, well-trained encoder, which makes its potential not limited to both tasks and experimental settings. Considering your concern, we would like to add above discussion in the 'Limitation and Future Work' section to inspire more thinking.

___
>**Q4:A very relevant work [1] is missing.**
We appreciate your careful reading, and now have supplemented the discussion of FICC [1] in the related work. Thank you!

___
[1]Become a Proficient Player with Limited Data through Watching Pure Videos. ICLR 2023
[2]Curl: Contrastive unsupervised representations for reinforcement learning. ICML 2020
[3]Reinforcement learning with prototypical representations. ICML 2021
[4]Masked world models for visual control. CoRL 2023
[5]Mask-based Latent Reconstruction for Reinforcement Learning. Neurips 2022
[6]Dreamerpro: Reconstruction-free model-based reinforcement learning with prototypical representations. ICML 2022




















## R3
We are deeply grateful to the reviewer for your careful reading, constructive reviews, and strong recommendation! We have further enhanced our paper in light of your feedback by supplementing more experiments and discussions!
___
>**Q1&Weakness2：Stage 3. (as per the summary above), in which the latent action model is finetuned and the action decoder is trained, requires real actions to be observed via environment interaction. It is unclear what is the impact of the schedule of environment interactions, how exactly the phases are optimally alternated so as to sample iteratively from the policy learned online via the concurrent step 4. Does this matter?**

The BCV-LR online stage contains three parts. First, we (a) allow the agent to interact with the environment for a fixed number of times using its policy and collect transitions to enrich the experience buffer. Immediately after that, (b) we perform finetuning of the latent action and training of the action decoder on the experience buffer. Then, (c) we train the latent policy to imitate the finetuned latent action predicted from expert videos. After this, the BCV-LR policy is improved and we return to part a) to collect better training data, which forms a cyclic online policy learning. This alternation is intuitive, that is, first collect new data of higher quality, then use the better data to obtain the better latent actions from expert videos, and finally let the policy learn the better latent actions to achieve performance improvement. You can refer to Sec.A.1 (Appendix) for concrete pseudo code of BCV-LR and we will further clarify the descriptions in the methodology section to make them more comprehensible to readers.

**For the impact of the schedule of environment interactions.** In previous experiments, we did not adjust the number of interactions for each cycle, but kept it fixed at a relatively large value from start to finish (for example, we fixed the number of collected transitions in part (a) as 1000 in DMControl) and ultimately achieved satisfactory results. To answer your question, we conduct additional experiments, where we try a much smaller interaction number (1000->2) and correspondingly reduce the number of update times for latent actions (100->1) and latent policies (1000->2) in each interaction, making BCV-LR perform in a fashion akin to off-policy RL. We denote this variant as 'BCV-LR(1000->2)'. The results in Table I show that BCV-LR can still achieve effective learning, which demonstrates that BCV-LR is robust to the schedule of environment interactions.

**Table I**
| task  | BCV-LR(1000->2) |  BCV-LR |   DrQv2| / |video|
| - | - | - | - | - | - | 
| reacher_hard |  875 ± 65 |  **900 ± 31**  | 92 ± 98  | | 967 |
| finger_spin  | **956 ± 20**  | 942 ± 48  | 374 ± 264 || 981|
___
>**Q2&Weakness4: How does the method compare to performing the steps in a fashion akin to LAPA? (No online learning, only imitation learning alignment on ground truth expert action?**

To answer your questions, we further conduct additional experiments, where we let BCV-LR perform latent action finetuning and policy imitation with a few action-labeled expert transitions. Concretely, we maintain the original pre-training stage and use 10k offline action-labeled expert transitions to achieve offline latent action finetuning and policy cloning with the original losses. The results in Table II (RL denotes DrQv2 for DMControl and PPO for procgen) demonstrate that BCV-LR can also achieve offline imitation learning alignment well if expert actions are provided. Of course, we would like to say that BCV-LR is designed for the ILV (imitation learning from videos), a much harder variant of the classical ILO (imitation learning from observation only) problem, where the employment of unsupervised online policy training without accessing expert actions adheres to the norms of this field.

**Table II**
|| task  | BCV-LR-offline |  BCV-LR |  RL | / |video|
|-| - | - | - | - | - | - | 
|DMControl| reacher_hard |  **938 ± 44** |  900 ± 31  | 92 ± 98  | | 967 |
|| finger_spin  | **978 ± 7**  | 942 ± 48  | 374 ± 264 || 981|
|Procgen| fruitbot  | **27.7 ± 0.4**  | 27.5 ± 1.5  | -1.9 ± 1.0 || 29.9|
___

>**Weakness3: The method as requested does require a source of ground truth actions from the videos, it is thus not strictly suitable for solving pressing hard problems for real world robotics, e.g. learning from real human videos and then transferring the policies to robots. This is also related to the benchmarks being limited to classic "toy RL" simulation environments.**

We agree with your comment! Since it is highly challenging to balance video imitation performance and efficiency without access to expert actions or expert rewards, our experiments have primarily been conducted in relatively standard visual RL environments to answer this open question, and we have not extended our work to real-world tasks or considered large-scale cross-domain pre-training, which are of greater significance. Taking your comment into account, we further conduct some extra experiments in Metaworld manipulation benchmark, which may demonstrate a wider application of BCV-LR for robots. 

For each Metaworld task, only 50k environmental steps are allowed, with remaining settings similar to that of DMControl. Results are shown in Table III. In this interaction-limited situation, BCV-LR can still derive effective manipulation skills from expert videos without accessing expert actions and rewards, which demonstrates its wider range of applications and potential for generalizing to real-world manipulation tasks. Furthermore, we would like to add a discussion on real-world tasks and cross-domain knowledge transferring in the 'Limitation and Future Work' sections to make our paper more comprehensive.

**Table III**
|Metaworld-50k|BCV-LR|BCO|DrQv2|/|video|
|-|-|-|-|-|-| 
|Faucet-open|**0.82 ± 0.20**|0.13 ± 0.19|0.00 ± 0.00||1|
|Reach|**0.63 ± 0.25**|0.03 ± 0.05|0.13 ± 0.12||1|
|Drawer-open|**0.92 ± 0.12**|0.13 ± 0.09|0.00 ± 0.00||1|
|Faucet-close|**0.98 ± 0.04**|0.00 ± 0.00|0.50 ± 0.28 ||1|
|Mean SR|**0.84**|0.07|0.16||1|






















## R4
We greatly appreciate the reviewer's careful reading and strong recognition of our work! We have carefully reviewed your comments, addressed your questions, and further improved our paper based on your feedback!
___
>**Q1：How general is the learned encoder? Is the shift image + contrastive loss learn useful features for non-video games, say real world robotics tasks? Or will this only work for relatively simplistic videos/environments?**

We apply this ‘shift image + contrastive loss’ with reconstruction (Eq.1, line 185, main paper) for Procgen video games because it has been proven effective in these kinds of tasks [1,2]. As shown in Sec.3.1.1 (line 174, main paper), BCV-LR is easily compatible with any action-free self-supervised tasks, and it can adapt to different types of domains by choosing appropriate self-supervised objectives. This motivates us to choose another prototype-based temporal association loss (Eq.10, line 549, Appendix) for DMControl environments where the temporal understanding is crucial and this loss has been proven better than ‘shift image + contrastive loss’ in DMControl [3]. To this end, BCV-LR can involve more advanced self-supervised objectives (e.g., ViT-based masked reconstruction [4,5]) for more challenging tasks (e.g., real-world manipulation) if necessary. In addition, BCV-LR can also be combined with an off-the-shelf, well-trained encoder, which makes its potential not limited to only video game environments.
___

>**Q2:How well does the mapping from latent action to action work in more complex environments, e.g., robot controls rather than just video game controls?**

In addition to the video games, we also demonstrate the advantages of BCV-LR on DMControl benchmark which consists of several continuous robotic control tasks. We summarize the average results below. More details are provided in Sec.4.3 (main paper) and Sec.C.1 (Appendix).

**Table I**
| DMControl-8-tasks-100k  | BCV-LR  | LAIFO   | BCO  |UPESV |TACO  | DrQv2| / |video|
| - | - | - | - | - | - |  - | - | - |
| Mean Score | **604**   | 158  | 336  | 18     | 310  | 232 | | 698|
| Video-norm Mean Score  | **0.78**   | 0.20  | 0.31  | 0.03     | 0.45  | 0.34 | | 1.00|


In addition, we further conduct additional experiments in the Metaworld manipulation benchmark. Only 50k environmental steps are allowed for each Metaworld task, with remaining settings similar to that of DMControl. Results are shown in Table II. In this interaction-limited situation, BCV-LR can still derive effective manipulation skills from expert videos without accessing expert actions and rewards, which demonstrates its wider range of applications and potential for generalizing to real-world manipulation tasks. Furthermore, we would like to add a discussion about real-world tasks in the 'Limitation and Future Work' sections to make our paper more comprehensive.

**Table II**
|Metaworld-50k|BCV-LR|BCO|DrQv2|/|video|
|-|-|-|-|-|-| 
|Faucet-open|**0.82 ± 0.20**|0.13 ± 0.19|0.00 ± 0.00||1|
|Reach|**0.63 ± 0.25**|0.03 ± 0.05|0.13 ± 0.12||1|
|Drawer-open|**0.92 ± 0.12**|0.13 ± 0.09|0.00 ± 0.00||1|
|Faucet-close|**0.98 ± 0.04**|0.00 ± 0.00|0.50 ± 0.28 ||1|
|Mean SR|**0.84**|0.07|0.16||1|

___
[1]Become a Proficient Player with Limited Data through Watching Pure Videos. ICLR 2023
[2]Curl: Contrastive unsupervised representations for reinforcement learning. ICML 2020
[3]Reinforcement learning with prototypical representations. ICML 2021
[4]Masked world models for visual control. CoRL 2023
[5]Mask-based Latent Reconstruction for Reinforcement Learning. Neurips 2022


