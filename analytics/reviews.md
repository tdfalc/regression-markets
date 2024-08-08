# Review 1

## Summary:
This paper tackles the challenge of data replication, a scenario where strategic agents exploit their data and assume false identities to augment their revenue. It introduces a replication-robust analytics market, exploring the nuances in representing analytics tasks as a cooperative game through different methodologies. Each representation is shown to have causal implications that affect the market's robustness against replication. The efficacy of the proposed approach is confirmed through validation with real-world experiments.

* Soundness: 3: good
* Presentation: 2: fair
* Contribution: 2: fair

## Strengths:
The considered topic is interesting to me. Theoretical proofs are abundant, and real-world experiments have demonstrated the effectiveness of the method.

## Weaknesses:
(1) The paper is hard to follow. Some equations are proposed without giving sufficient explanations; 
(2) The innovation of this paper is not clearly articulated; is it the proposal of an algorithm? This aspect is not explicitly highlighted in this paper. 
(3) If not only replication attacks but also malicious data attacks are considered, can the method proposed in this paper solve the problem? 
(4) The concept of game theory is mentioned in this paper. It would be better if the replication behavior could be explicitly explained as a best response from the perspective of the utility function.

## Questions:
See weaknesses.

## Limitations:
The authors have adequately addressed the limitations of their work.

## Flag For Ethics Review: No ethics review needed.
Rating: 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Code Of Conduct: Yes

# Review 2

## Summary:
Data (analytics) markets offer a compelling solution for using data from diverse sources, including competitors. The prevailing approach rewards data providers with payments proportional to the value their data brings to the analytics task. However, a critical flaw undermines this framework: the ease with which data can be replicated creates an incentive for agents to duplicate their data and assume false identities, leading to grossly unfair payments. This paper introduces a novel analytics market design, leveraging Shapley value attribution to counter spiteful agents and ensure robustness against replication. Additionally, the authors propose a comprehensive market design that integrates existing solutions and explores various representations of analytics tasks as cooperative games.

* Soundness: 3: good
* Presentation: 2: fair
* Contribution: 3: good

## Strengths:
The paper addresses an important problem of replication robustness in the analytics markets. Similar to prior works they also use Shapley value based attribution but they adapt it with the interventional lift to make it robust to agents potentially replicating the data to game the system.

The authors showed that the use of observational lift to value a coalition of agents is the source of the vulnerability to replication in the market and they argue that the prior works trying to solve this using penalization methods can only achieve weak robustness. In contrast, their proposal based on interventional lift can provide strict robustness to replication.

Experimental results on wind power forecasting data show with the proposed attribution method the payments remain constant irrespective of the replication. This validates their theory.

## Weaknesses:
(1) The paper can be improved in writing and presentation. It is hard to describe the core of the solution in simple language but it would be greatly helpful in understanding the paper. 
(2) There are several places with abstract phrases that are hard to understand without more information, e.g. the abstract, what is meant by “causal nuances”, in other places what is the “full causal picture”? 
(3) The abstract does not convey the heart of the solution. It would be useful to tell what exactly makes your solution robust instead of saying the second last line. 
(4) Section 3, comes up as a surprise, please provide some motivation and roadmap in the earlier sections towards this. 
(5) More than the mathematical details I’d appreciate why and how these lifts apply to the setting and how the interventional lift ensures robustness.
(6) Is replication in the experiments with zero variance? Experiments on more datasets and having more than one agent with replication would make the paper stronger.

## Questions:
See above.

## Limitations:
Yes.

## Flag For Ethics Review: No ethics review needed.
Rating: 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Code Of Conduct: Yes

# Review 3
## Summary:
The paper introduces an analytics market based on Shapley value attribution that is replication-robust and applies it to a close-to-real-life case of wind power forecasting. While the theoretical result is quite elegant and straightforward, the example looks redundant and only weakly related to real life. If I understood correctly, a group of wind turbine companies can perfectly know the number of features available for the market, so replication of data is hardly possible. The example looks quite artificial. Note that SHAP is computationally hard, so a high-dimensional example would be more interesting given the presented theorem.

* Soundness: 4: excellent
* Presentation: 3: good
* Contribution: 3: good

## Strengths:
The paper is well written: the problem statement, methods, and contribution are all clear. The proposed problem looks important for the domain and the solution was proposed (also illustrated by an example).

## Weaknesses:
It is unclear how scalable is the proposed solution with an increasing number of agents/observations/features. The proposed example is quite a toy contrary to all exaggerated statements on the relevance to real-life. Also, there is no feedback from the revenue distribution to the actual willingness to share the data, so, in principle, one might say that there is no problem here. The data will be shared anyway.

## Questions:
It is unclear how scalable is the proposed solution with an increasing number of agents/observations/features. The proposed example is quite a toy contrary to all exaggerated statements on the relevance to real-life. 

Pls, provide theoretical analysis for analytical solutions and empirical results for approximate methods for larger problems. 

It would be nice to know when software/RAM bottlenecks kick in to clarify the practical value for realistic problems.

 Consider close to the actual number of wind turbines of actual companies for a rather big region.


## Limitations:
pls, see Questions and Weakness

## Flag For Ethics Review: No ethics review needed.
Rating: 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Code Of Conduct: Yes