# BCV-LR

___
>**Q1：How general is the learned encoder? Is the shift image + contrastive loss learn useful features for non-video games, say real world robotics tasks? Or will this only work for relatively simplistic videos/environments?**

We apply this ‘shift image + contrastive loss’ (Eq.1, Line 185, Mainpaper) for Procgen video games because it has been proven effective in these kinds of tasks. As shown in Sec.3.1.1 (Line 174, Mainpaper), BCV-LR is easily compatible with any action-free self-supervised tasks and it can adapt to different types of domains by choosing appropriate self-supervised objectives. This motivates us to choose another prototype-based temporarl loss (Eq.10, Line 549, Appendix) for Deepmind control tasks in our experiments, where the temporarl understanding is crucial for agents and this temporarl loss has been proven better than contrastive loss in previous works. To this end, BCV-LR can involve more advanced self-supervised objective (e.g., ViT-based masked reconstruction) for more challenging tasks (e.g., real-world tasks) if necessary. In addition, BCV-LR can also be combined with an off-the-shelf well-trained encoder, which makes its potential not limited to only video game environments.
___

>**Q2:How well does the mapping from latent action to action work in more complex environments, e.g., robot controls rather than just video game controls?**

In addition to the video games, we also demonstrate the advantages of BCV-LR on DMControl benchmark which consists of several continuous robotic control tasks. We summarize the average results here in Table I . More detailed results are provided in  

In addition, we further conducted additional experiments in Metaworld manipulation benchmark. Only 50k environmental steps are allowed for each Metaworld task, with remaining settings similar to that of DMControl. Results are shown in Table 2. In this interaction-limited situation, BCV-LR can still derive effective manipulation skills from expert videos without accessing expert actions and rewards, which demonstrates its potential for generalizing to real-world manipulation tasks.

| Task   | η=0.02   | η=0.1    | η=0.5    | adaptive η | DVFB    |
| - | - | - | - | - | - |
| Stand  | 954±5    | 951±10   | 961±8    | 960±5      | 965±7   |
| Walk   | 753±32   | 765±18   | 752±31   | 820±77     | 908±21  |

___

