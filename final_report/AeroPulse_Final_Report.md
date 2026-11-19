# AeroPulse
## Flight Delay Prediction and Operational Intelligence

**Project type:** End-to-end machine learning and analytics  
**Domain:** U.S. domestic aviation  
**Development data:** August 2026  
**External validation:** September 2026  
**Final classifier:** XGBoost  
**Probability calibration:** Sigmoid  
**Decision threshold:** 0.253331  

---

## Executive Overview

AeroPulse investigates U.S. flight departure delays
using historical airline operational data.

The project combines data preprocessing, feature
engineering, exploratory analysis, model comparison,
probability calibration, threshold optimization,
external temporal validation, and robustness analysis.


## 1. Final External Validation

The final model is an XGBoost classifier trained on
August 2026 flight records and evaluated on an
independent September 2026 dataset.

The September evaluation covers **563,591 flights**,
with an observed delay prevalence of **22.81%**.

The model achieved:

- ROC-AUC: **0.7142**
- PR-AUC: **0.4249**
- F1-score: **0.4588**
- Brier score: **0.1577**
- Log loss: **0.4861**

A decision threshold of **0.253331** was
selected using August validation data and then
applied to September without re-optimizing it
on the September outcomes.

These results describe predictive performance,
not a causal explanation of flight delays.



## 2. Tree Model Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.3400 | 0.6479 | 0.3450 | 0.6113 | 0.4411 | 0.6913 | 0.3835 |
| XGBoost | 0.3800 | 0.7028 | 0.3901 | 0.5462 | 0.4551 | 0.7108 | 0.4103 |



## 3. Probability Calibration

| Method | Brier | LogLoss | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- |
| Original XGBoost | 0.1667 | 0.5056 | 0.7108 | 0.4103 |
| Sigmoid | 0.1582 | 0.4869 | 0.7108 | 0.4103 |
| Isotonic | 0.1584 | 0.4876 | 0.7108 | 0.4053 |



## 4. Threshold Evaluation

| Dataset | Threshold | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| August Validation | 0.2533 | 0.7031 | 0.3865 | 0.6047 | 0.4716 | 0.7246 | 0.4230 | 0.1516 |
| September External | 0.2533 | 0.7094 | 0.3988 | 0.5401 | 0.4588 | 0.7142 | 0.4249 | 0.1577 |
| September Threshold 0.50 | 0.5000 | 0.7793 | 0.5944 | 0.1018 | 0.1738 | 0.7142 | 0.4249 | 0.1577 |



## 5. Operational Findings

- **Departure period:** Evening had the highest observed delay rate (34.62%), while Morning had the lowest (12.95%).
- **Airline sensitivity:** Among airlines with at least 10,000 flights, B6 had the highest recall (76.43%), while MQ had the lowest (27.31%).
- **Destination exposure:** Among destinations with at least 10,000 flights, BOS had the highest observed delay rate (42.51%).



## 6. Airline Robustness

| Reporting_Airline | Flights | ActualDelayRate | Precision | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| WN | 109894.0000 | 0.3164 | 0.5073 | 0.7502 | 0.6053 | 0.1777 |
| DL | 84617.0000 | 0.1694 | 0.3213 | 0.3033 | 0.3120 | 0.1330 |
| AA | 79814.0000 | 0.2697 | 0.4276 | 0.5650 | 0.4867 | 0.1813 |
| UA | 70782.0000 | 0.2004 | 0.3359 | 0.4666 | 0.3906 | 0.1501 |
| OO | 70563.0000 | 0.1814 | 0.3292 | 0.3429 | 0.3359 | 0.1409 |
| AS | 28839.0000 | 0.1820 | 0.2856 | 0.3906 | 0.3299 | 0.1431 |
| YX | 28571.0000 | 0.2031 | 0.3533 | 0.4329 | 0.3891 | 0.1499 |
| MQ | 27788.0000 | 0.1903 | 0.3180 | 0.2731 | 0.2939 | 0.1496 |
| OH | 19287.0000 | 0.2152 | 0.3010 | 0.6742 | 0.4162 | 0.1657 |
| B6 | 18503.0000 | 0.2580 | 0.3649 | 0.7643 | 0.4940 | 0.1758 |
| F9 | 17925.0000 | 0.2423 | 0.3594 | 0.6418 | 0.4608 | 0.1693 |
| G4 | 7008.0000 | 0.1908 | 0.3341 | 0.4473 | 0.3825 | 0.1450 |



## 7. Departure Period Robustness

| DepPeriod | Flights | ActualDelayRate | Precision | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Morning | 241941.0000 | 0.1295 | 0.3600 | 0.1050 | 0.1626 | 0.1079 |
| Afternoon | 170453.0000 | 0.2737 | 0.3878 | 0.5615 | 0.4588 | 0.1872 |
| Evening | 136876.0000 | 0.3462 | 0.4159 | 0.8048 | 0.5484 | 0.2082 |
| Night | 14321.0000 | 0.2223 | 0.3168 | 0.5712 | 0.4076 | 0.1653 |



## 8. Hourly Robustness

| ScheduledDepHour | Flights | ActualDelayRate | Recall | F1 |
| --- | --- | --- | --- | --- |
| 7.0000 | 42282.0000 | 0.0936 | 0.0217 | 0.0409 |
| 8.0000 | 42056.0000 | 0.1147 | 0.0433 | 0.0779 |
| 10.0000 | 38577.0000 | 0.1728 | 0.1377 | 0.2002 |
| 6.0000 | 37974.0000 | 0.0750 | 0.0028 | 0.0055 |
| 17.0000 | 36352.0000 | 0.3234 | 0.7697 | 0.5200 |
| 11.0000 | 35787.0000 | 0.2073 | 0.2260 | 0.2783 |
| 12.0000 | 35619.0000 | 0.2275 | 0.3606 | 0.3611 |
| 13.0000 | 35087.0000 | 0.2498 | 0.4616 | 0.4134 |
| 15.0000 | 33924.0000 | 0.2985 | 0.6609 | 0.4933 |
| 18.0000 | 33673.0000 | 0.3561 | 0.8056 | 0.5548 |
| 14.0000 | 33125.0000 | 0.2824 | 0.5550 | 0.4618 |
| 9.0000 | 33003.0000 | 0.1442 | 0.0828 | 0.1331 |
| 16.0000 | 32698.0000 | 0.3148 | 0.7126 | 0.5096 |
| 19.0000 | 27681.0000 | 0.3622 | 0.8292 | 0.5598 |
| 20.0000 | 23093.0000 | 0.3738 | 0.8480 | 0.5839 |
| 21.0000 | 16077.0000 | 0.3099 | 0.7612 | 0.5172 |
| 5.0000 | 12262.0000 | 0.0705 | 0.0012 | 0.0023 |
| 22.0000 | 9159.0000 | 0.2474 | 0.6222 | 0.4427 |
| 23.0000 | 4145.0000 | 0.1865 | 0.5058 | 0.3378 |
| 0.0000 | 697.0000 | 0.1263 | 0.1591 | 0.1628 |



## 9. Origin Airport Robustness

| Origin | Flights | ActualDelayRate | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- |
| ORD | 31805.0000 | 0.2952 | 0.6396 | 0.4868 | 0.1961 |
| DEN | 26878.0000 | 0.2539 | 0.6363 | 0.5072 | 0.1629 |
| DFW | 26189.0000 | 0.2760 | 0.4849 | 0.4721 | 0.1869 |
| ATL | 25233.0000 | 0.2029 | 0.4404 | 0.3782 | 0.1512 |
| LAX | 15340.0000 | 0.1941 | 0.3584 | 0.3801 | 0.1438 |
| SEA | 14571.0000 | 0.2141 | 0.5131 | 0.3995 | 0.1553 |
| CLT | 14333.0000 | 0.2230 | 0.6158 | 0.4250 | 0.1639 |
| PHX | 14190.0000 | 0.2156 | 0.4812 | 0.4497 | 0.1508 |
| LAS | 13442.0000 | 0.2482 | 0.6760 | 0.5055 | 0.1611 |
| SFO | 12774.0000 | 0.2683 | 0.6711 | 0.4970 | 0.1748 |
| DCA | 11596.0000 | 0.2591 | 0.6648 | 0.4746 | 0.1784 |
| BOS | 11471.0000 | 0.2827 | 0.5964 | 0.5412 | 0.1791 |
| MCO | 10854.0000 | 0.2277 | 0.7155 | 0.4908 | 0.1530 |
| LGA | 10483.0000 | 0.2687 | 0.6571 | 0.5167 | 0.1725 |
| MSP | 9733.0000 | 0.1557 | 0.2805 | 0.2929 | 0.1256 |



## 10. Destination Airport Robustness

| Dest | Flights | ActualDelayRate | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- |
| ORD | 31757.0000 | 0.2511 | 0.4852 | 0.3945 | 0.1848 |
| DEN | 26783.0000 | 0.2020 | 0.5814 | 0.4622 | 0.1399 |
| DFW | 26153.0000 | 0.2018 | 0.3427 | 0.3668 | 0.1513 |
| ATL | 25121.0000 | 0.1705 | 0.4093 | 0.3681 | 0.1299 |
| LAX | 15366.0000 | 0.1991 | 0.4890 | 0.4363 | 0.1418 |
| SEA | 14571.0000 | 0.1815 | 0.3888 | 0.3718 | 0.1363 |
| CLT | 14364.0000 | 0.1821 | 0.4371 | 0.3800 | 0.1391 |
| PHX | 14191.0000 | 0.2346 | 0.5191 | 0.4900 | 0.1574 |
| LAS | 13450.0000 | 0.2474 | 0.6763 | 0.5169 | 0.1582 |
| SFO | 12776.0000 | 0.3063 | 0.8244 | 0.5267 | 0.1925 |
| BOS | 11597.0000 | 0.4251 | 0.5897 | 0.6045 | 0.2457 |
| DCA | 11594.0000 | 0.2296 | 0.6480 | 0.4540 | 0.1627 |
| MCO | 10875.0000 | 0.2382 | 0.6579 | 0.4873 | 0.1594 |
| LGA | 10451.0000 | 0.2708 | 0.6226 | 0.4981 | 0.1774 |
| MSP | 9750.0000 | 0.1639 | 0.3304 | 0.3414 | 0.1260 |



## 11. Daily Robustness

| FlightDate | Flights | ActualDelayRate | Recall | F1 |
| --- | --- | --- | --- | --- |
| 2026-09-24 | 20469.0000 | 0.2280 | 0.5997 | 0.4820 |
| 2026-09-27 | 20308.0000 | 0.3092 | 0.6887 | 0.5723 |
| 2026-09-20 | 20294.0000 | 0.3361 | 0.6699 | 0.5807 |
| 2026-09-13 | 20262.0000 | 0.2679 | 0.6922 | 0.5247 |
| 2026-09-25 | 20130.0000 | 0.2382 | 0.6437 | 0.5015 |
| 2026-09-03 | 20129.0000 | 0.3728 | 0.5668 | 0.5970 |
| 2026-09-17 | 20113.0000 | 0.3512 | 0.5702 | 0.5785 |
| 2026-09-14 | 20084.0000 | 0.2229 | 0.6062 | 0.4462 |
| 2026-09-04 | 20002.0000 | 0.3355 | 0.5897 | 0.5576 |
| 2026-09-18 | 20001.0000 | 0.2763 | 0.6220 | 0.5284 |
| 2026-09-28 | 19978.0000 | 0.2395 | 0.6231 | 0.4839 |
| 2026-09-07 | 19872.0000 | 0.2124 | 0.6615 | 0.4717 |
| 2026-09-10 | 19673.0000 | 0.2130 | 0.5594 | 0.4308 |
| 2026-09-11 | 19389.0000 | 0.2531 | 0.6281 | 0.5067 |
| 2026-09-21 | 19265.0000 | 0.3110 | 0.6382 | 0.5810 |
| 2026-09-30 | 18434.0000 | 0.1623 | 0.3215 | 0.3134 |
| 2026-09-23 | 18196.0000 | 0.1477 | 0.3292 | 0.3050 |
| 2026-09-16 | 18161.0000 | 0.1769 | 0.3378 | 0.3435 |
| 2026-09-29 | 18041.0000 | 0.1320 | 0.2632 | 0.2228 |
| 2026-09-02 | 17914.0000 | 0.1795 | 0.3520 | 0.3600 |
| 2026-09-15 | 17863.0000 | 0.2085 | 0.3147 | 0.3379 |
| 2026-09-08 | 17791.0000 | 0.1548 | 0.3312 | 0.3012 |
| 2026-09-09 | 17677.0000 | 0.1413 | 0.2886 | 0.2583 |
| 2026-09-22 | 17661.0000 | 0.1891 | 0.2812 | 0.2892 |
| 2026-09-06 | 17595.0000 | 0.1567 | 0.6740 | 0.3550 |
| 2026-09-01 | 17450.0000 | 0.1574 | 0.3582 | 0.3324 |
| 2026-09-26 | 17007.0000 | 0.1322 | 0.4807 | 0.2829 |
| 2026-09-12 | 16763.0000 | 0.1784 | 0.4702 | 0.3373 |
| 2026-09-05 | 16760.0000 | 0.1906 | 0.4975 | 0.3755 |
| 2026-09-19 | 16309.0000 | 0.2729 | 0.4799 | 0.4457 |



## 12. Risk Calibration by Probability Band

| RiskBand | Flights | ActualDelayRate | MeanPredictedRisk | CalibrationGap |
| --- | --- | --- | --- | --- |
| (-0.001, 0.1] | 138338 | 0.0833 | 0.0702 | -0.0131 |
| (0.1, 0.2] | 179121 | 0.1671 | 0.1457 | -0.0214 |
| (0.2, 0.3] | 121820 | 0.2661 | 0.2463 | -0.0198 |
| (0.3, 0.4] | 70019 | 0.3757 | 0.3442 | -0.0315 |
| (0.4, 0.5] | 32278 | 0.4734 | 0.4440 | -0.0295 |
| (0.5, 0.6] | 16566 | 0.5757 | 0.5440 | -0.0317 |
| (0.6, 0.7] | 5247 | 0.6505 | 0.6356 | -0.0149 |
| (0.7, 0.8] | 201 | 0.6716 | 0.7160 | 0.0444 |
| (0.8, 0.9] | 1 | 1.0000 | 0.8060 | -0.1940 |



## 13. Category Drift

| Feature | UnseenRows | UnseenRate |
| --- | --- | --- |
| Reporting_Airline | 0 | 0.0000 |
| Origin | 57 | 0.0001 |
| Dest | 60 | 0.0001 |
| DepPeriod | 0 | 0.0000 |


## 14. Model Integrity Audit

- Passed checks: **22**
- Failed checks: **0**
- Review flags: **3**
- Automated audit status: **REVIEW**

### Review flags

- **Training procedure verification**: Date partitions are valid, but the saved model alone cannot prove which rows were used for fitting.
- **Original category encoding provenance**: August-wide sorted categories reproduce the validation convention, but original training category order must be checked in 07_tree_models.py.
- **Original calibration input provenance**: Logit input reproduces the convention in scripts 13 and 14. Verify original fitting transformation in 10b_probability_calibration.py.

The original training and calibration source code was
also inspected separately. It confirms that XGBoost
was fitted on the August 1-20 training partition and
that sigmoid calibration used the logit transformation.

Automated audit flags are retained for transparency.

## 15. Methodological Limitations


1. **Temporal coverage:** The study uses August 2026
   for model development and September 2026 for
   external temporal validation. Longer-term and
   seasonal generalization remains untested.

2. **Validation reuse:** Sigmoid calibration was fitted
   on the August validation partition, which was also
   used for threshold optimization. Consequently,
   validation-based threshold performance should not
   be interpreted as a fully independent estimate.

3. **Category handling:** Native categorical XGBoost
   was trained with training-partition category levels.
   Category compatibility and handling of unseen
   September values require careful interpretation.

4. **Class imbalance:** Flight delays are less frequent
   than on-time departures. Precision, recall, PR-AUC,
   and threshold behavior are therefore particularly
   important.

5. **Operational trade-off:** A lower classification
   threshold improves delay detection but may increase
   false alarms.

6. **No causal claims:** Airline, airport, and time-of-day
   differences are descriptive associations and do not
   establish causal effects.

7. **Flight identity:** The saved prediction file was
   checked for date and target alignment, but unique
   flight identifiers were not available for a stronger
   row-level identity audit.


## 16. Reproducibility

The project includes Python scripts for data
preparation, feature engineering, machine learning,
external validation, robustness analysis, and
model integrity auditing.

Saved prediction probabilities were reproduced with
a maximum reported difference of 0.0000000000.

Reproducibility within the saved pipeline does not
by itself establish generalization to other months,
airlines, airports, or operational conditions.

## 17. Conclusion

AeroPulse demonstrates an end-to-end workflow for
predicting departure delays and examining how model
performance varies across operational segments.

Independent September evaluation provides a more
credible estimate of out-of-time performance than
the reused August validation partition.

The final system is a research and portfolio prototype,
not a production-ready airline decision system.
