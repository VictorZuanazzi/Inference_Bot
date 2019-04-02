# Inference_Bot
Learning sentence representations from natural language inference data


The first practical of the Statistical Methods for Natural Language Semantics course concerns learning general-purpose sentence representations in the natural language inference (NLI) task. The goal of this practical is threefold: to implement three neural models to classify sentence pairs based on their relation, to train these models using the Stanford Natural Language Inference (SNLI)
corpus [1] and evaluate the trained models using the SentEval framework [2].

NLI is the task of classifying entailment or contradiction relationships between premises and hypotheses, such as the following:

1. Premise Bob is in his room, but because of the thunder and lightning outside, he cannot sleep.
2. Hypothesis 1 Bob is awake.
3. Hypothesis 2 It is sunny outside.

While the first hypothesis follows from the premise, indicated by the alignment of ‘cannot sleep’ and ‘awake’, the second hypothesis contradicts the premise, as can be seen from the alignment of ‘sunny’ and ‘thunder and lightning’ and recognizing their incompatibility.
