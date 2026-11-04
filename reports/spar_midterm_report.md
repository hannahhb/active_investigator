Active Investigator: Can Follow-up Questions Catch Missing Reasoning?
Hannah Bansal
drvkbansal@gmail.com

Abstract

Chain-of-thought (CoT) monitors only work when the reasoning they need is actually written down. I study a model organism that takes shortcuts from misleading documentation and skips the reasoning that would expose them. I built an investigator that reads the model's CoT, writes targeted follow-up questions, and asks the model to answer them with the documentation removed. One well-chosen question gets the model to contradict its own wrong answer about three times as often as asking it to "re-check" (52% vs 19%). A simple trained selector beats the LLM's own pick (61% vs 52%), but both remain well below the best question in hindsight (89%, with no false alarms).

Introduction

When a model gives an answer without the reasoning that should support it, a passive monitor has very little to read. This matters for oversight because the failures we care about most are the ones where the model never writes down the step that would give it away. I want to know whether an investigator that actively asks targeted follow-up questions can surface that missing evidence more reliably and cheaply than a passive judge or a simple "reconsider your answer" prompt.

My research question is: can follow-up questions designed only from the model's own CoT (no access to execution traces) get a shortcut-following model to reveal that its answer was not earned?

Contributions so far:
1. A trace-free investigator framework in which the investigator, judge and model under test are interchangeable, so a weaker investigator can later be swapped in to test weak-to-strong monitoring.
2. An evidence-level definition of a "catch" that does not reward an investigator for simply solving the task itself.
3. Baselines (passive judge, re-check prompt, LLM-picked and random questions) and a first trained question selector.

Related Work

Guan et al. [1] show that CoT monitorability depends on how much reasoning is externalised, and that follow-up questions can raise it; my organism is an extreme case where the CoT is close to empty. Khan et al. [2] find that debate between stronger models helps weaker judges reach the truth, which motivates letting a weak investigator interrogate a stronger model. Pacchiardi et al. [3] detect lies in black-box LLMs by asking follow-up questions, but their questions are fixed and unrelated to the task; mine are targeted at the specific step that was skipped. AI control [4] and weak-to-strong generalisation [5] frame the end goal: trusted, weaker monitors overseeing stronger, untrusted models.

Methods

Model organism: PyINE's Qwen3-4B "shortcut" model [6], which predicts a Python program's output. Tasks come in three variants: original, hinted (the docstring has a correct example) and misleading (the docstring has a wrong example). I use the TACO guardrail test pool (611 problems), taking one record per problem and variant, and the organism is served with vLLM.

Protocol (per record):
1. Investigate: the investigator (gpt-5-mini) reads the program, input, CoT and answer, gives a passive trust score, and writes 5 ranked follow-up questions, each with the value the original answer implies.
2. Ask: the organism answers each question in a fresh conversation with docstrings and comments removed.
3. Re-check baseline: in the original conversation, the organism is asked to check its answer once more.
4. Judge: gpt-5-mini labels each reply as supporting or contradicting the original answer.

Selection arms (all computed from the same replies): the investigator's top-k, random-k, a logistic-regression selector (question type, rank, wording, CoT length; no labels or variant) cross-fitted over 5 problem folds, and a hindsight oracle.

Metrics: AUROC of the trust score, and evidence-level catch: a wrong answer counts as caught only if the organism's own reply contradicts it. The false-alarm rate is the same flag on correct answers.

Results

The data is 1,830 records from all 611 problems; 765 answers are wrong (520 misleading, 227 original, 18 helpful). Every method is scored on the same records, with one question per record unless noted.

Table 1 reports evidence-level catches in the PyINE format: a wrong answer counts as caught only when the organism's own follow-up reply contradicts it, so an investigator that merely re-solves the task gets no credit. Figure 1 plots catch against false alarms (A) and by task variant (B).

- Re-checking rarely works: the organism changes only 19% of its wrong answers.
- One targeted question catches 52–61% of wrong answers at 7–10% false alarms, roughly three times re-checking (+33 points, 95% CI +29 to +37).
- The investigator's own first question is no better than a random one. The trained selector beats both: +9 points over the LLM's pick (95% CI +6 to +13) and +7 over random (+4 to +9).
- Questions work best on misleading tasks (Δmislead +11 to +14 points) and rarely on helpful ones, whose 18 errors do not come from the docstring.
- All five questions catch 89%, but with 32% false alarms; the oracle's single question catches 89% with almost none (0.3%).
- AUROC barely separates the methods (0.89–0.91, against 0.90 for the passive judge), which is why the evidence-level catch is the main metric.

Discussion

The main takeaway is that the evidence is there to be elicited. When the organism is made to do the step it skipped, without the misleading docstring, it usually gets it right and contradicts itself. The bottleneck is choosing which question to ask. The hindsight oracle shows that a single good question would catch 89% of wrong answers with no false alarms, yet a strong LLM tends to lead with "safe" setup questions (parsing the input, early variables) that the organism gets right whether or not its final answer is correct.

One surprise was how well the passive gpt-5-mini judge does (AUROC 0.90). This is because it can solve many of these programs itself, which is exactly why I report evidence-level catches: they do not reward the investigator for re-solving the task, so they stay meaningful once the investigator is weaker than the model it monitors.

Limitations: there is a single investigator model and seed; the judge's verdicts are noisy (I found missed and invented contradictions when reading transcripts); about 7% of follow-up replies hit the token limit; and hinted-wrong cases are too rare to analyse (18 of 1,830 records).

Next steps:
1. Run weak-to-strong: swap in investigators and judges smaller than the organism (e.g. 3–4B models) and test whether evidence-level catches hold when passive judging fails.
2. Improve selection: prompt for questions on the value that decides the output and on the docstring's specific claim, use structural features, and train a pairwise ranker on the full 611-problem run.
3. Add a rule-based check of each reply against the implied value, to reduce judge noise.

Conclusion

I built a trace-free active investigator that designs follow-up questions from a model's chain of thought, and an evidence-level way to score it. On a shortcut-prone code-execution organism, targeted questions expose unearned answers far more often than asking the model to re-check. The open problem, and the focus of the second half, is learning which question to ask, especially with a weaker investigator.

References

    [1] M. Y. Guan, M. Wang, M. Carroll, Z. Dou, A. Y. Wei, M. Williams, B. Arnav, J. Huizinga, I. D. Kivlichan, A. Glaese, J. Pachocki and B. Baker, "Monitoring Monitorability," in Proc. 43rd Int. Conf. Machine Learning (ICML), PMLR, vol. 306, pp. 37538–37602, 2026. arXiv:2512.18311.
    [2] A. Khan, J. Hughes, D. Valentine, L. Ruis, K. Sachan, A. Radhakrishnan, E. Grefenstette, S. R. Bowman, T. Rocktäschel and E. Perez, "Debating with More Persuasive LLMs Leads to More Truthful Answers," in Proc. 41st Int. Conf. Machine Learning (ICML), PMLR, vol. 235, 2024. arXiv:2402.06782.
    [3] L. Pacchiardi, A. J. Chan, S. Mindermann, I. Moscovitz, A. Y. Pan, Y. Gal, O. Evans and J. Brauner, "How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions," in Proc. 12th Int. Conf. Learning Representations (ICLR), 2024. arXiv:2309.15840.
    [4] R. Greenblatt, B. Shlegeris, K. Sachan and F. Roger, "AI Control: Improving Safety Despite Intentional Subversion," in Proc. 41st Int. Conf. Machine Learning (ICML), PMLR, vol. 235, 2024. arXiv:2312.06942.
    [5] C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet, M. Joglekar, J. Leike, I. Sutskever and J. Wu, "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision," in Proc. 41st Int. Conf. Machine Learning (ICML), PMLR, vol. 235, 2024. arXiv:2312.09390.
    [6] P.-L. St-Charles, A. Palmas, D. Fornasiere, S. Lei, M. Bronzi, J.-P. Falet, I. Serban and Y. Bengio, "PyINE: A Framework for Scalable Elicitation and Oversight via Code Execution," LawZero, technical report v1.1, 2026. https://saifh-github.github.io/pyine/
