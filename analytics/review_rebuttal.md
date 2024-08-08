# Global 
Comment to Everybody: The authors would like to thank all reviewers for the careful handling of our manuscript, as well as the time and effort invested in evaluating our work. We are glad they found the contributions to be of value to the community. 

Below, we address the questions raised by each reviewer. We hope to have addressed all concerns and look forward to further discussion if needed.

# Review 1
Thank you for reviewing our paper. We appreciate your interest in the topic and are glad that you found the theoretical proofs and experimental analyses to effectively support our proposal. We address your questions and concerns below:

(1) We appreciate your feedback on the clarity of our paper. To enhance readability, we have significantly improved the explanations surrounding our equations, providing additional context and illustrative examples in the appendix.

(2) The core innovation of our paper is the use of the interventional conditional expectation when calculating the Shapley value, as opposed to the observational approach in existing works related to revenue allocation in analytics markets. We have strengthened the paper by explicitly highlighting this contribution in the introduction and related work sections. We have also included an algorithmic description of the revenue allocation, outlining the change to this procedure that we propose.

(3) Thank you for raising this important point. While our focus in this paper is on replication attacks, we acknowledge the significance of malicious data attacks. We believe that our proposed method can be extended to address malicious data attacks by [briefly outline potential extensions or modifications]. A comprehensive investigation into the robustness of our method against such attacks is an interesting avenue for future work.

(4) We agree that explicitly framing replication behavior as a best response to a utility function would strengthen our paper. Specifically, a seller's utility can be modelled as revenue earned in the market minus costs (e.g., collection costs, privacy loss, etc.). We have included a clear definition of this utility function and how replication emerges as a strategic optimal choice. 


# Review 2
Thank you for the detailed feedback. We try to address the questions and comments below:

(1) We appreciate your feedback on the clarity of our paper. We agree that improving the paper's accessibility is crucial. We've addressed this by enhancing the overall readability and providing a more intuitive explanation of our proposal. In addition, we have included a detailed algorithmic description the revenue allocation procedure to explicitly highlight the change from existing works. 

(2) Thank you for pointing out the abstract language in our paper. We recognize the importance of clear communication and have revised the manuscript accordingly. We have replaced abstract terms like "causal nuances" and "full causal picture" with more explicit and informative language. For example, "causal nuances" refers to the distinctions between conditioning by observation or intervetnion from a causal perspective, and "full causal picture" is simply the description of how these lifts impact the causal relationships amongst features in the market. 

(3) We agree that the abstract could more clearly convey the core of our proposal. We have revised the abstract to explicitly highlight the key novelties of our approach and how they contribute to its robustness. The abstract now reads as follows:

```Despite the widespread use of machine learning throughout industry, many firms face a common challenge: relevant datasets are typically distributed amongst market competitors that are reluctant to share information. We develop an analytics markets provide monetary incentives for data sharing with application to supervised learning problems. To allocate revenue, we treat features of agents as players in a cooperative game, which requires simulating loss evaluation for each coalition. We show that existing works take an observational approach to model the distribution of out-of-coalition features, which exposes the market to malicous behaviour, where agents replicate their data and act under false identities to increase revenue, and in turn diminish that of others. Instead, we use an interventional approach which we show to be replication-robust by design. Our methodology is validated using a real-world wind power forecasting case study.```

(4) We appreciate your feedback on the flow of our paper and agree that Section 3 would benefit from additional context. To address this, we will revise the preceding sections to enhance the overall narrative. Specifically, we will strengthen the introduction, refine the roadmap, and restructure Section 2 to more clearly demonstrate how the groundwork laid there motivates the lift formulations in Section 3, the choice of which affects the properties the market, as detailed in Section 4.

(5) By treating features as players in a cooperative game, we can calculate the expected marginal contribution of each to the loss function using the Shapley value. This requires evaluating the loss function on each subset (coalition) of features. However, without re-training, machine learning models typically require a value for each input feature. The question is, to evaluate the loss on each coalition of features, what should be the distribution of out-of-coalition features? This is where the two lifts come in. We consider two methods to model this distrubtion, namely taking the observational and interventional conditional expectations. 

The observational approach will spread credit equally amongst correlated features, even if the caudal effects are only indirect on the target, hence if the owner of one of these correlated features replicates their data many times and submits these to the market under false identities, their overall share of the revenue will increase. However, the interventional approach only considers direct causal effects on the target, therefore any replicates (which have only indirect effects by design) will not be credited, thereby ensuring robustness to replication. 

To clarify these points within the manuscript, we have improved the explanations surrounding our equations, providing additional context and illustrative examples in the appendix.

(6) In our experimental analysis, the dataset is fixed and the model parameters have analytic solutions so the results have zero variance. While we appreciate the suggestion to broaden our experiments through additional datasets, we respectfully contend that this would not contribute to the paper's strength. Specifically, unlike proposals of novel learning algorithms where dataset diversity serves as a valuable proxy for generalization, the interventional lift we study is inherently replication-robust in general, thereby independent of specific datasets. 

Furthermore, the benefit of using the interventional lift is not dependent on the number of agents replicating, it is more to remove the dominant strategy to replicate their features altogether. For example, consider the example in Figure 3 where we have two identical features owned by different agents. In this example, we showed that if one agent replciates their feature k times, this agent will get an increasing proportion of the revenue with increasing k when using the observational lift. However, suppose instead that both agents replciate their feature k times. In this case, both agents will have still have equal proportions of the revenue, hence their overall revenue will be unchanged even as k increases. In this example, the unfair revenue allocation only comes about with the number of replciations of each agent is asymmetric. As a result, the dominant strategy of each agent is simply to replicate their feature infinitely many times, which is infeasible in practice for both the agents and for the market itself to deal with. The interventional lift, removes replication as a dominant strategy, has there is no possibility to increase revenue even if the number of replications is asymmetric. Perhaps extending our experimental anlyses to show this phenomenon would strength the paper?


# Review 3
(1) Thank you for pointing out this important detail. In the paper, we write that the Shapley value can be expressed in terms of permutations such that
$$
\phi_{i} = \frac{1}{|\mathcal{I}_{-c}|!} \sum_{\theta \in \Theta} \Big( v(\mathcal{I}_{c} \cup \{j : j \prec_\theta i\}) -  v(\mathcal{I}_{c} \cup \{j : j \preceq_\theta i\})\Big), \, \forall i \in \mathcal{I}_{-c},
$$
where the set $\mathcal{C}$ represents a coalition in the cooperative game. Evaluation of the Shapley value is known to be NP-hard in general, and is thus an open challenge in the data valuation and wider machine learning communities. For instance, many works (e.g., [1--3]) develop sampling-based approximation
methods, albeit for different applications. For instance, a simple approach would be to use a Monte-Carlo estimator, whereby
$$
\hat{\phi}_{i} = \frac{1}{|\mathcal{P}|} \sum_{\theta \in \mathcal{P}} \Big( v(\mathcal{I}_{c} \cup \{j : j \prec_\theta i\}) -  v(\mathcal{I}_{c} \cup \{j : j \preceq_\theta i\})\Big), \, \forall i \in \mathcal{I}_{-c},
$$
for a uniform sample of permutations $\mathcal{P} \subset \Theta$. This provides an unbiased estimator, which, by the Central Limit Theorem, converges asymptotically at a rate of $\mathcal{O}(1/\sqrt{|\mathcal{P}|})$. However, observe that in both cases, the expression inside the parenthesis remains the same, as evaluation of the characteristic function local to a particular coalition. Therefore, the replication robustness property holds exactly even for approximate methods. This is not neceassrily the case for other market properties, for example budget balance, for which one would need to assess wheter this estimator violates the properties or provide boiunds the approximation error thereof. Whilst this is an interesting avenue for future work, we respectfully contend that empirical assessment of larger problems is necessary, since we have shown our proposed property holds regardless of wheter an anltyic or approximate method is used, hence holds generally for all problem sizes.










Also, there is no feedback from the revenue distribution to the actual willingness to share the data, so, in principle, one might say that there is no problem here. The data will be shared anyway.


