# References - IEEE-CIS Fraud Detection (LightGBM + Heterogeneous GNN)

Verified against CrossRef (api.crossref.org/works/{doi}) in May 2026. Each entry below resolved to the listed authors and title. Volume / issue / page numbers are intentionally omitted to avoid fabrication; author, title, journal or venue, year, and DOI are sufficient for a primary-source lookup.

## Gradient boosting and tabular ML foundations

1. **Breiman, L.** Random Forests. *Machine Learning*. 2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324). Bagged decision trees with random feature subsetting; the canonical baseline for tabular fraud detection and the reference implementation behind feature-importance reports.

2. **Friedman, J. H.** Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*. 2001. DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451). Original gradient-boosting framework. The mathematical foundation behind XGBoost, LightGBM, and CatBoost.

3. **Chen, T., Guestrin, C.** XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*. 2016. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785). The dominant tabular-ML system on Kaggle through 2018-2021 and a top-3 finisher on the IEEE-CIS leaderboard in ensemble form.

4. **Arik, S. O., Pfister, T.** TabNet: Attentive Interpretable Tabular Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*. 2021. DOI: [10.1609/aaai.v35i8.16826](https://doi.org/10.1609/aaai.v35i8.16826). Deep tabular network with sequential attention; directly comparable to gradient boosting on the IEEE-CIS task.

5. **Lundberg, S. M., Erion, G., Chen, H., et al.** From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*. 2020. DOI: [10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9). Polynomial-time TreeSHAP for boosted ensembles. The default interpretability layer over LightGBM fraud models.

## Class imbalance and resampling

6. **Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P.** SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*. 2002. DOI: [10.1613/jair.953](https://doi.org/10.1613/jair.953). The reference oversampling method for imbalanced binary classification.

7. **Han, H., Wang, W.-Y., Mao, B.-H.** Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning. *Lecture Notes in Computer Science*. 2005. DOI: [10.1007/11538059_91](https://doi.org/10.1007/11538059_91). Variant restricting synthetic samples to the decision boundary; useful when card-level fraud clusters are dense in feature space.

8. **He, H., Bai, Y., Garcia, E. A., Li, S.** ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. *2008 IEEE International Joint Conference on Neural Networks*. 2008. DOI: [10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969). Adaptive density-based synthetic sampling.

9. **He, H., Garcia, E. A.** Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*. 2009. DOI: [10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239). The standard survey on imbalance algorithms; framework for the cost-sensitive evaluation in this project.

10. **Krawczyk, B.** Learning from Imbalanced Data: Open Challenges and Future Directions. *Progress in Artificial Intelligence*. 2016. DOI: [10.1007/s13748-016-0094-0](https://doi.org/10.1007/s13748-016-0094-0). Modern survey covering deep-learning era extensions of resampling and cost-sensitive learning.

11. **Fernandez, A., Garcia, S., Herrera, F., Chawla, N. V.** SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-Year Anniversary. *Journal of Artificial Intelligence Research*. 2018. DOI: [10.1613/jair.1.11192](https://doi.org/10.1613/jair.1.11192). Anniversary review summarising 15 years of SMOTE variants and benchmarks.

12. **Buda, M., Maki, A., Mazurowski, M. A.** A Systematic Study of the Class Imbalance Problem in Convolutional Neural Networks. *Neural Networks*. 2018. DOI: [10.1016/j.neunet.2018.07.011](https://doi.org/10.1016/j.neunet.2018.07.011). Empirical study of imbalance handling in CNNs; methodology transfers to the GNN advanced model.

13. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P.** Focal Loss for Dense Object Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 2020. DOI: [10.1109/TPAMI.2018.2858826](https://doi.org/10.1109/TPAMI.2018.2858826). Focal-loss formulation; used as the GNN training objective in `src/model_advanced.py`.

## Cost-sensitive learning and evaluation

14. **Provost, F., Fawcett, T.** Robust Classification for Imprecise Environments. *Machine Learning*. 2001. DOI: [10.1023/A:1007601015854](https://doi.org/10.1023/A:1007601015854). ROC convex hull and cost-sensitive operating-point selection; theoretical foundation for the cost-weighted score reported here.

15. **Fawcett, T.** An Introduction to ROC Analysis. *Pattern Recognition Letters*. 2006. DOI: [10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010). The standard tutorial on ROC, AUC, and operating-point selection.

16. **Davis, J., Goadrich, M.** The Relationship Between Precision-Recall and ROC Curves. *Proceedings of the 23rd International Conference on Machine Learning*. 2006. DOI: [10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874). Why PR-AUC is more honest than ROC-AUC under heavy imbalance; the IEEE-CIS prevalence (~3.5%) sits squarely in the regime where this matters.

## Credit-card and payment fraud detection

17. **Ngai, E. W. T., Hu, Y., Wong, Y. H., Chen, Y., Sun, X.** The Application of Data Mining Techniques in Financial Fraud Detection: A Classification Framework. *Decision Support Systems*. 2011. DOI: [10.1016/j.dss.2010.08.006](https://doi.org/10.1016/j.dss.2010.08.006). Classification framework over a decade of fraud-detection data-mining literature.

18. **Bhattacharyya, S., Jha, S., Tharakunnel, K., Westland, J. C.** Data Mining for Credit Card Fraud: A Comparative Study. *Decision Support Systems*. 2011. DOI: [10.1016/j.dss.2010.08.008](https://doi.org/10.1016/j.dss.2010.08.008). Side-by-side comparison of logistic regression, SVM, and random forest on real card-fraud data.

19. **Whitrow, C., Hand, D. J., Juszczak, P., Weston, D., Adams, N. M.** Transaction Aggregation as a Strategy for Credit Card Fraud Detection. *Data Mining and Knowledge Discovery*. 2008. DOI: [10.1007/s10618-008-0116-z](https://doi.org/10.1007/s10618-008-0116-z). Foundation of the rolling-aggregation feature blocks (count, sum, mean over windows) used in `src/model_baseline.py`.

20. **Bahnsen, A. C., Aouada, D., Stojanovic, A., Ottersten, B.** Feature Engineering Strategies for Credit Card Fraud Detection. *Expert Systems with Applications*. 2016. DOI: [10.1016/j.eswa.2015.12.030](https://doi.org/10.1016/j.eswa.2015.12.030). Time-aware aggregations and periodic features; directly motivates the time-of-day, day-of-week, and time-since-last-tx features.

21. **Dal Pozzolo, A., Caelen, O., Johnson, R. A., Bontempi, G.** Calibrating Probability with Undersampling for Unbalanced Classification. *2015 IEEE Symposium Series on Computational Intelligence*. 2015. DOI: [10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33). Probability calibration after undersampling; required when the baseline is consumed by a downstream Bayesian risk-pricing layer.

22. **Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., Bontempi, G.** Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. *IEEE Transactions on Neural Networks and Learning Systems*. 2018. DOI: [10.1109/TNNLS.2017.2736643](https://doi.org/10.1109/TNNLS.2017.2736643). Production-realistic evaluation protocol (delayed labels, concept drift, verification budget) that informs the Discussion section.

23. **Carcillo, F., Le Borgne, Y.-A., Caelen, O., Kessaci, Y., Oble, F., Bontempi, G.** Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection. *Information Sciences*. 2021. DOI: [10.1016/j.ins.2019.05.042](https://doi.org/10.1016/j.ins.2019.05.042). Hybrid pipeline that blends anomaly detection with supervised classification; relevant when the GNN model is reused as an anomaly-style scorer.

24. **Fiore, U., De Santis, A., Perla, F., Zanetti, P., Palmieri, F.** Using Generative Adversarial Networks for Improving Classification Effectiveness in Credit Card Fraud Detection. *Information Sciences*. 2019. DOI: [10.1016/j.ins.2017.12.030](https://doi.org/10.1016/j.ins.2017.12.030). GAN-based oversampling alternative to SMOTE for credit-card fraud.

## Adaptive boosting

25. **Freund, Y., Schapire, R. E.** A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. *Journal of Computer and System Sciences*. 1997. DOI: [10.1006/jcss.1997.1504](https://doi.org/10.1006/jcss.1997.1504). The AdaBoost paper; historical anchor for boosting-based fraud detectors.

## Graph neural networks for fraud and anomaly detection

26. **Hu, Z., Dong, Y., Wang, K., Sun, Y.** Heterogeneous Graph Transformer. *Proceedings of The Web Conference 2020*. 2020. DOI: [10.1145/3366423.3380027](https://doi.org/10.1145/3366423.3380027). HGT architecture with type-aware attention; the alternative GNN backbone evaluated in `src/model_advanced.py`.

27. **Liu, Y., Ao, X., Qin, Z., Chi, J., Feng, J., Yang, H., He, Q.** Pick and Choose: A GNN-Based Imbalanced Learning Approach for Fraud Detection. *Proceedings of the Web Conference 2021*. 2021. DOI: [10.1145/3442381.3449989](https://doi.org/10.1145/3442381.3449989). PC-GNN: explicit imbalance-aware sampling for GNN fraud detectors. Direct methodological reference for the advanced model.

28. **Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., Yu, P. S.** Enhancing Graph Neural Network-Based Fraud Detectors against Camouflaged Fraudsters. *Proceedings of the 29th ACM International Conference on Information and Knowledge Management*. 2020. DOI: [10.1145/3340531.3411903](https://doi.org/10.1145/3340531.3411903). CARE-GNN; tackles the case where fraudsters mimic legitimate users; relevant to defensive evaluation of the IEEE-CIS GNN.

29. **Akoglu, L., Tong, H., Koutra, D.** Graph Based Anomaly Detection and Description: A Survey. *Data Mining and Knowledge Discovery*. 2014. DOI: [10.1007/s10618-014-0365-y](https://doi.org/10.1007/s10618-014-0365-y). Survey of graph-anomaly methods that pre-date GNNs; useful for situating the GNN advanced model against classical alternatives.

30. **Pang, G., Shen, C., Cao, L., Hengel, A. v. d.** Deep Learning for Anomaly Detection: A Review. *ACM Computing Surveys*. 2021. DOI: [10.1145/3439950](https://doi.org/10.1145/3439950). Modern review of deep anomaly methods; benchmarks the autoencoder and one-class neural baselines that the GNN must outperform.
