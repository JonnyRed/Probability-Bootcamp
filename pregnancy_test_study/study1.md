# Pregnancy study

"To evaluate the accuracy of a pregnancy test facility using the provided
ase study involving:

* 1,000 women
  * 500 pregnant and
  * 500 not pregnant
  
We can set up a framework for testing the accuracy of the pregnancy tests.
The accuracy of a pregnancy test can be assessed using the following metrics:

* True Positives (TP): The number of pregnant women who test positive 
for pregnancy.
* True Negatives (TN): The number of non-pregnant women who test negative 
for pregnancy.
* False Positives (FP): The number of non-pregnant women who test positive 
for pregnancy.
* False Negatives (FN): The number of pregnant women who test negative 
for pregnancy.
  
Steps to Evaluate the Accuracy
Conduct the Tests: Administer the pregnancy tests to all 1,000 women.

Collect Results: Record the results in a confusion matrix format:

|               | Pregnant (Actual) | Not Pregnant (Actual) |
|---------------|-------------------|-----------------------|
| Test Positive | TP                | FP                    |
| Test Negative | FN                | TN                    |

Calculate Metrics:

* Sensitivity (True Positive Rate): Measures the proportion of actual 
positives that are correctly identified.
$\text{Sensitivity} = \frac{TP}{TP + FN}$
* Specificity (True Negative Rate): Measures the proportion of actual 
negatives that are correctly identified. 
$\text{Specificity} = \frac{TN}{TN + FP}$
* Positive Predictive Value (PPV): Measures the proportion of positive 
test results that are true positives. 
$\text{PPV} = \frac{TP}{TP + FP}$
* Negative Predictive Value (NPV): Measures the proportion of negative 
test results that are true negatives. 
$\text{NPV} = \frac{TN}{TN + FN}$
* Accuracy: Measures the overall correctness of the test. 
$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

Example Scenario
Let's assume the following results after testing:

* TP = 450 (pregnant women who tested positive)
* TN = 480 (non-pregnant women who tested negative)
* FP = 20 (non-pregnant women who tested positive)
* FN = 50 (pregnant women who tested negative)

Using these values, we can calculate the metrics:

* Sensitivity: $\text{Sensitivity} = \frac{450}{450 + 50} = \frac{450}{500} = 0.90 \text{ or } 90%$
* Specificity: $\text{Specificity} = \frac{480}{480 + 20} = \frac{480}{500} = 0.96 \text{ or } 96%$
* Positive Predictive Value (PPV): $\text{PPV} = \frac{450}{450 + 20} = \frac{450}{470} \approx 0.957 \text{ or } 95.7%$
* Negative Predictive Value (NPV): $\text{NPV} = \frac{480}{480 + 50} = \frac{480}{530} \approx 0.906 \text{ or } 90.6%$
* Accuracy: $\text{Accuracy} = \frac{450 + 480}{1000} = \frac{930}{1000} = 0.93 \text{ or } 93%$

#### Conclusion

By analyzing the results, we can conclude that the pregnancy test facility
has a high sensitivity (90%), specificity (96%), and overall accuracy (93%).
These metrics provide a comprehensive view of the test's performance and
can help in assessing its reliability in clinical settings.
