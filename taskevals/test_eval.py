from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

import os

    


# Your inputs
question = "How do they perform multilingual training?"
answer = 'In the paper, multilingual training is carried out through a shared auxiliary decoder and joint training across languages. The system adds an auxiliary task of predicting the morphosyntactic description (MSD) of the target form, and the parameters of this MSD decoder are shared across multiple languages. Instead of grouping languages strictly by family (which would leave some languages isolated, like Russian), the authors randomly group sets of two to three languages. During training, minibatches are drawn from different languages in random order, so the model alternates between them without being told explicitly which language the data comes from. This is based on the assumption that abstract morphosyntactic features can be shared across languages. After this multilingual pretraining phase (20 epochs), the model undergoes monolingual fine-tuning for each language individually, with a reduced learning rate. This ensures that the multilingual knowledge is retained while still adapting specifically to each language. The results show that this multilingual training strategy improves accuracy substantially—on average, multilingual models outperform monolingual ones by about 7.96%, and the combination of multilingual training with fine-tuning yields the best results for most languages' 
context = """
"The parameters of the entire MSD (auxiliary-task) decoder are shared across languages."
"We experiment with random groupings of two to three languages."
"Multilingual training is performed by randomly alternating between languages for every new minibatch."
"We do not pass any information to the auxiliary decoder as to the source language…"
"After 20 epochs of multilingual training, we perform 5 epochs of monolingual finetuning for each language."
"…we reduce the learning rate to a tenth of the original learning rate, i.e. 0.0001…"
"…multilingual results [are] 7.96% higher than monolingual ones on average."
"Monolingual finetuning improves accuracy across the board… by 2.72% on average."
"…the multi-tasking approach paired with multilingual training and subsequent monolingual finetuning outperforms… for five out of seven languages."
"""
expected_output = 'Multilingual training is performed by randomly alternating between languages for every new minibatch'

# question = 'How do they demonstrate that this type of EEG has discriminative information about the intended articulatory movements responsible for speech?'
# answer = 'They demonstrate that this type of EEG contains discriminative information about the articulatory movements underlying speech by showing both quantitative performance gains and qualitative visual evidence. The authors designed a hierarchical deep learning framework combining CNNs, LSTMs, and a deep autoencoder, followed by gradient boosting for classification. When tested on imagined speech EEG data from the KARA ONE dataset, their mixed model achieved an average accuracy of 77.9% across five binary phonological classification tasks, which represents a 22.5% improvement over earlier methods. These results indicate that EEG signals encode reliable cues about articulatory processes like nasal vs. non-nasal sounds, bilabial consonants, and vowel distinctions  . Beyond raw accuracy, they also presented t-SNE visualizations of the learned feature spaces. These plots showed clear separation between different phonological categories (e.g., nasal vs. non-nasal, vowel vs. consonant), providing visual confirmation that their method extracted meaningful articulatory patterns from EEG. Finally, they reported that their approach not only improved average accuracy but also reduced variability across subjects, suggesting more robust and generalizable representations of the articulatory information embedded in EEG signals'
# context = """
#         “Our best models achieve an average accuracy of 77.9% across five different binary classification tasks, providing a significant 22.5% improvement over previous methods. As we also show visually, our work demonstrates that the speech imagery EEG possesses significant discriminative information about the intended articulatory movements responsible for natural speech synthesis.”

#         “From the average accuracy scores, we observe that the mixed network performs much better than individual blocks… The very fact that our combined network improves the classification accuracy by a mean margin of 14.45% than the CNN-LSTM network indeed reveals that the autoencoder contributes towards filtering out the unrelated and noisy features… In addition to accuracy, we also provide the kappa coefficients… A higher mean kappa value corresponding to a task implies that the network is able to find better discriminative information from the EEG data beyond random decisions. … To further investigate the feature representation achieved by our model, we plot T-distributed Stochastic Neighbor Embedding (tSNE)… The tSNE visualization reveals that the second set of features are more easily separable than the first one, thereby giving a rationale for our performance.

# """
# Build test case
test_case = LLMTestCase(
    input=question,
    actual_output=answer,
    expected_output=expected_output,
    retrieval_context=[context]
)

# Define metrics
metrics = [
    AnswerRelevancyMetric(),
    FaithfulnessMetric(),
    ContextualPrecisionMetric(),
    ContextualRecallMetric(),
]

# Run metrics one by one
for metric in metrics:
    score = metric.measure(test_case)
    print(f"{metric.__class__.__name__}: {score}")
