from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

print("Matrix sklearn.metrics:", confusion_matrix(
    df.actual_label.values, df.predicted_RF.values))


def find_TP(y_true, y_pred):
    # Кількість істинних позитивних (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    # Кількість хибних негативних (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    # Кількість хибних позитивних (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    # Кількість істинних негативних (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


# Функція для обчислення усіх 4 значень
def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

# Функція для створення матриці плутанини


def pavlenko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


print("Matrix Pavlenko:", pavlenko_confusion_matrix(
    df.actual_label.values, df.predicted_RF.values))
print("------------------------------------------------")

assert np.array_equal(pavlenko_confusion_matrix(df.actual_label.values, df.predicted_RF.values), confusion_matrix(
    df.actual_label.values, df.predicted_RF.values)), 'my_confusion_matrix() is not correct for RF'
assert np.array_equal(pavlenko_confusion_matrix(df.actual_label.values, df.predicted_LR.values), confusion_matrix(
    df.actual_label.values, df.predicted_LR.values)), 'my_confusion_matrix() is not correct for LR'


print("accuracy_score RF sklearn.metrics:", accuracy_score(
    df.actual_label.values, df.predicted_RF.values))
print("accuracy_score LR sklearn.metrics:", accuracy_score(
    df.actual_label.values, df.predicted_LR.values))
print("")


def pavlenko_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


# Перевірка точності для моделі RF
assert pavlenko_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(
    df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'

# Перевірка точності для моделі LR
assert pavlenko_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(
    df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'

# Вивід точності
print('Accuracy RF Pavlenko:', pavlenko_accuracy_score(
    df.actual_label.values, df.predicted_RF.values))
print('Accuracy LR Pavlenko:', pavlenko_accuracy_score(
    df.actual_label.values, df.predicted_LR.values))
print("------------------------------------------------")


print("recall_score RF sklearn.metrics:", recall_score(
    df.actual_label.values, df.predicted_RF.values))
print("recall_score LR sklearn.metrics:", recall_score(
    df.actual_label.values, df.predicted_LR.values))
print("")


def pavlenko_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Уникнути ділення на нуль
    return recall


    # Перевірка recall для моделі RF
assert pavlenko_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(
    df.actual_label.values, df.predicted_RF.values), 'my_recall_score failed on RF'

# Перевірка recall для моделі LR
assert pavlenko_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(
    df.actual_label.values, df.predicted_LR.values), 'my_recall_score failed on LR'

# Вивід recall
print('Recall RF Pavlenko: ', pavlenko_recall_score(
    df.actual_label.values, df.predicted_RF.values))
print('Recall LR Pavlenko: ', pavlenko_recall_score(
    df.actual_label.values, df.predicted_LR.values))
print("------------------------------------------------")


print("precision_score RF sklearn.metrics:", precision_score(
    df.actual_label.values, df.predicted_RF.values))
print("precision_score LR sklearn.metrics:", precision_score(
    df.actual_label.values, df.predicted_LR.values))
print("")


def pavlenko_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    # Уникнути ділення на нуль
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision


# Перевірка precision для моделі RF
assert pavlenko_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(
    df.actual_label.values, df.predicted_RF.values), 'my_precision_score failed on RF'

# Перевірка precision для моделі LR
assert pavlenko_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values, df.predicted_LR.values), 'my_precision_score failed on LR'

# Вивід precision
print('Precision RF Pavlenko: ', pavlenko_precision_score(
    df.actual_label.values, df.predicted_RF.values))
print('Precision LR Pavlenko: ', pavlenko_precision_score(
    df.actual_label.values, df.predicted_LR.values))

print("------------------------------------------------")

print("f1_score RF sklearn.metrics:", f1_score(
    df.actual_label.values, df.predicted_RF.values))
print("f1_score LR sklearn.metrics:", f1_score(
    df.actual_label.values, df.predicted_LR.values))
print("")


def pavlenko_f1_score(y_true, y_pred):
    recall = pavlenko_recall_score(y_true, y_pred)
    precision = pavlenko_precision_score(y_true, y_pred)
    f1 = (2 * precision * recall) / (precision + recall) if (precision +
          recall) > 0 else 0  # Уникнути ділення на нуль
    return f1


# Перевірка F1 для моделі RF
assert pavlenko_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(
    df.actual_label.values, df.predicted_RF.values), 'my_f1_score failed on RF'

# Перевірка F1 для моделі LR
# assert pavlenko_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'my_f1_score failed on LR'

# Вивід F1
print('F1 RF Pavlenko: ', pavlenko_f1_score(
    df.actual_label.values, df.predicted_RF.values))
print('F1 LR Pavlenko: ', pavlenko_f1_score(
    df.actual_label.values, df.predicted_LR.values))

f1_my = pavlenko_f1_score(df.actual_label.values, df.predicted_LR.values)
f1_sklearn = f1_score(df.actual_label.values, df.predicted_LR.values)
print(f'My F1: {f1_my}, Sklearn F1: {f1_sklearn}')
print("------------------------------------------------")


print('scores with threshold = 0.5')
print('Accuracy RF: % .3f'% (pavlenko_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (pavlenko_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: % .3f'% (pavlenko_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (pavlenko_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')

print('scores with threshold = 0.25')
print('Accuracy RF: % .3f'% (pavlenko_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (pavlenko_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: % .3f'% (pavlenko_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (pavlenko_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print("------------------------------------------------")


from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

