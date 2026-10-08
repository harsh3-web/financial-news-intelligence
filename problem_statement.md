# Problem Statement

## 1. What the system does
the system helps an analyst quickly sort large volumes of financial news by whether each statement is good, bad, or neutral news for the company. It does not predict prices.

## 2. Why financial sentiment is harder than general sentiment
1.same word can have different sentiment depends on the context(e.g. reduced word in "profit reduced by 20%" and "cost reduced by 20%")
2. Numbers carry the meaning: "Sales rose to EUR 5 mn from EUR 3 mn" has no emotional words at all.
Most sentences are neutral/factual, so the model must separate "fact" from "good/bad news".
3. Words have different meaning in finance than their actual meaning
..like "liability", "bull", or "crude" (as in crude oil, not rude).

## 3. Research questions
- RQ1:How do classic Ml,rnn,and pretrained transformer compare in Macro F-1 on financial sentence classification?

RQ2: IS LSTM and GRU outperform than the simpeRNN and Is complex LSTM worth it over GRU?
RQ3: what is tradeoff between preditive performance and cost like parameters, training time, inference
RQ4: Which phenomena (negation, numbers, comparatives, mixed sentiment, domain terms) cause errors, and do more complex models fix them or just make different mistakes?

## 4. Primary metric
 my primary mmetric is macro F-1 becoz it treats all class equally so model cant score well just by pretending neutral.
 scondary is accuracy and presision and recall, accuracy is easy to communicate but it mislead under unbalanced datadet,negatice class matter because missing leading error is costly to the investors.
 diagnostic:confusion matrix-beczuse it shows which class get confused 
 cost- training parameter, training time,inference time - training parameters-model size and memory,training time-development cost,inference latency-deployment cost 

## 5. Fairness rules for the comparison
Macro-F1 is the average of the F1 scores of each class, computed separately. I chose it over accuracy because my dataset is imbalanced: most sentences are neutral. Accuracy rewards the majority class, so a model that predicts 'neutral' for everything could get high accuracy while being useless. For example, with 90 neutral, 5 positive and 5 negative sentences, that model scores 90% accuracy but only about 0.32 Macro-F1, because it gets zero F1 on positive and negative. Macro-F1 forces the model to do well on every class. The trade-off is that it weighs all classes equally, and treats all errors the same, even though calling negative news positive is worse than calling it neutral. That's why I also look at the confusion matrix and the negative-class recall.