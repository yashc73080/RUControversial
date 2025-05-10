# google/electra-base-discriminator:
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

Training with 5 epochs, early stopping with patience=2
Epoch 1/5 - Avg training loss: 1.4046
Validation - Accuracy: 0.2083, F1: 0.0784
New best model saved with F1: 0.0784
Unfreezing encoder parameters...
Epoch 2/5 - Avg training loss: 1.3751
Validation - Accuracy: 0.5413, F1: 0.5108
New best model saved with F1: 0.5108
Epoch 3/5 - Avg training loss: 1.3631
Validation - Accuracy: 0.2565, F1: 0.2434
Epoch 4/5 - Avg training loss: 1.3567
Validation - Accuracy: 0.3583, F1: 0.3964
Early stopping triggered after 4 epochs
Training complete.
**Test Accuracy: 0.5506**
Test F1-Score: 0.5158
                  precision    recall  f1-score   support

         asshole       0.26      0.26      0.26       837
  everyone sucks       0.45      0.02      0.04       216
no assholes here       0.17      0.07      0.10       379
 not the asshole       0.65      0.77      0.70      2513

        accuracy                           0.55      3945
       macro avg       0.38      0.28      0.28      3945
    weighted avg       0.51      0.55      0.52      3945


# roberta-base:
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

Epoch 1: avg train loss 1.5971
Val acc: 0.1262, F1: 0.0726
Saved best model.
Epoch 2: avg train loss 1.5454
Val acc: 0.1769, F1: 0.1516
Saved best model.
Epoch 3: avg train loss 1.4918
Val acc: 0.2129, F1: 0.2248
Saved best model.
Epoch 4: avg train loss 1.4037
Val acc: 0.3168, F1: 0.3342
Saved best model.
Epoch 5: avg train loss 1.3122
Val acc: 0.4126, F1: 0.4524
Saved best model.
**Test Acc: 0.4117**, F1: 0.4552
                  precision    recall  f1-score   support

         asshole       0.39      0.34      0.36       837
  everyone sucks       0.14      0.43      0.21       216
no assholes here       0.18      0.56      0.27       379
 not the asshole       0.76      0.41      0.54      2513

        accuracy                           0.41      3945
       macro avg       0.37      0.43      0.34      3945
    weighted avg       0.59      0.41      0.46      3945


# albert-base-v2
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

Connection successful!
Total number of samples: 19722
Class distribution: verdict
not the asshole     12562
asshole              4183
no assholes here     1897
everyone sucks       1080
Name: count, dtype: int64
Data preparation complete.
Train samples: 13804, Validation samples: 1973, Test samples: 3945
Class distribution in training set: [2928  756 1328 8792]
Class weights: [ 4.714481  18.25926   10.394578   1.5700637]
Using device: cuda
Model and optimizer initialized.
Training with 5 epochs, early stopping with patience=2
Epoch 1/5 - Avg training loss: 1.3904
Validation - Accuracy: 0.1956, F1: 0.1128
New best model saved with F1: 0.1128
Unfreezing encoder parameters...
Epoch 2/5 - Avg training loss: 1.3372
Validation - Accuracy: 0.4815, F1: 0.4871
New best model saved with F1: 0.4871
Epoch 3/5 - Avg training loss: 1.3278
Validation - Accuracy: 0.2940, F1: 0.3257
Epoch 4/5 - Avg training loss: 1.3226
Validation - Accuracy: 0.3512, F1: 0.3999
Early stopping triggered after 4 epochs
**got error and failed**

# microsoft/deberta-v3-base
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

Training with 5 epochs, early stopping with patience=2
Epoch 1/5 - Avg training loss: 1.4029
Validation - Accuracy: 0.2048, F1: 0.0738
New best model saved with F1: 0.0738
Unfreezing encoder parameters...
Epoch 2/5 - Avg training loss: 1.3884
Validation - Accuracy: 0.5702, F1: 0.5107
New best model saved with F1: 0.5107
Epoch 3/5 - Avg training loss: 1.3832
Validation - Accuracy: 0.2408, F1: 0.2679
Epoch 4/5 - Avg training loss: 1.3860
Validation - Accuracy: 0.4389, F1: 0.4516
Early stopping triggered after 4 epochs
Training complete.
**Test Accuracy: 0.5744**
Test F1-Score: 0.5116
                  precision    recall  f1-score   support

         asshole       0.26      0.22      0.24       837
  everyone sucks       0.00      0.00      0.00       216
no assholes here       0.00      0.00      0.00       379
 not the asshole       0.64      0.83      0.72      2513

        accuracy                           0.57      3945
       macro avg       0.23      0.26      0.24      3945
    weighted avg       0.46      0.57      0.51      3945

# microsoft/deberta-v3-small
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

Training with 5 epochs, early stopping with patience=2
Epoch 1/5 - Avg training loss: 1.4121
Validation - Accuracy: 0.2119, F1: 0.0741
New best model saved with F1: 0.0741
Unfreezing encoder parameters...
Epoch 2/5 - Avg training loss: 1.3869
Validation - Accuracy: 0.6356, F1: 0.5006
New best model saved with F1: 0.5006
Epoch 3/5 - Avg training loss: 1.3768
Validation - Accuracy: 0.2261, F1: 0.1088
Epoch 4/5 - Avg training loss: 1.3704
Validation - Accuracy: 0.4693, F1: 0.4851
Early stopping triggered after 4 epochs
Training complete.
Test Accuracy: 0.6370
Test F1-Score: 0.5048
                  precision    recall  f1-score   support

         asshole       0.45      0.02      0.05       837
  everyone sucks       0.00      0.00      0.00       216
no assholes here       0.00      0.00      0.00       379
 not the asshole       0.64      0.99      0.78      2513

        accuracy                           0.64      3945
       macro avg       0.27      0.25      0.21      3945
    weighted avg       0.50      0.64      0.50      3945

# xlm-roberta-base
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

Training with 5 epochs, early stopping with patience=2
Epoch 1/5 - Avg training loss: 1.4121
Validation - Accuracy: 0.2119, F1: 0.0741
New best model saved with F1: 0.0741
Unfreezing encoder parameters...
Epoch 2/5 - Avg training loss: 1.3869
Validation - Accuracy: 0.6356, F1: 0.5006
New best model saved with F1: 0.5006
Epoch 3/5 - Avg training loss: 1.3768
Validation - Accuracy: 0.2261, F1: 0.1088
Epoch 4/5 - Avg training loss: 1.3704
Validation - Accuracy: 0.4693, F1: 0.4851
Early stopping triggered after 4 epochs
Training complete.
Test Accuracy: 0.6370
Test F1-Score: 0.5048
                  precision    recall  f1-score   support

         asshole       0.45      0.02      0.05       837
  everyone sucks       0.00      0.00      0.00       216
no assholes here       0.00      0.00      0.00       379
 not the asshole       0.64      0.99      0.78      2513

        accuracy                           0.64      3945
       macro avg       0.27      0.25      0.21      3945
    weighted avg       0.50      0.64      0.50      3945

# distilroberta-base
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

**failed**

# abhishek/bert-base-uncased-reddit
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 3e-4, 'weight_decay': 0.01}
]

**Failed**


# Final:
train_deberta_small_20250508_125651.43548603.o

Connection successful!
Total number of samples: 19722
Class distribution: verdict
not the asshole     12562
asshole              4183
no assholes here     1897
everyone sucks       1080
Name: count, dtype: int64
Data preparation complete.
Train samples: 13804, Validation samples: 1973, Test samples: 3945
Class distribution in training set: [2928  756 1328 8792]
Class weights: [ 6.  25.  15.   1.5]
Using device: cuda
Model and optimizer initialized.
Training with 5 epochs, early stopping with patience=1.
Step 50: Loss = 7.4903, Accuracy = 0.4888
Step 100: Loss = 6.9780, Accuracy = 0.2650
Step 150: Loss = 6.7733, Accuracy = 0.2838
Step 200: Loss = 6.9214, Accuracy = 0.2075
Step 250: Loss = 6.5740, Accuracy = 0.2225
Step 300: Loss = 6.0298, Accuracy = 0.3600
Step 350: Loss = 6.6853, Accuracy = 0.2600
Step 400: Loss = 7.1688, Accuracy = 0.2075
Step 450: Loss = 7.1436, Accuracy = 0.1325
Step 500: Loss = 6.6950, Accuracy = 0.3425
Step 550: Loss = 7.1616, Accuracy = 0.1713
Step 600: Loss = 7.0306, Accuracy = 0.1588
Step 650: Loss = 6.6213, Accuracy = 0.2125
Step 700: Loss = 6.8145, Accuracy = 0.1875
Step 750: Loss = 7.0665, Accuracy = 0.1963
Step 800: Loss = 6.5133, Accuracy = 0.1862
Step 850: Loss = 6.9515, Accuracy = 0.1900
Epoch 1/5 - Avg training loss: 6.8579, Accuracy: 0.2390
Validation - Loss: 6.8352, Accuracy: 0.2119, F1: 0.0741
New best model saved with F1: 0.0741
Unfreezing encoder parameters...
Step 900: Loss = 6.3469, Accuracy = 0.1436
Step 950: Loss = 6.6801, Accuracy = 0.2375
Step 1000: Loss = 6.8039, Accuracy = 0.2412
Step 1050: Loss = 6.7417, Accuracy = 0.2412
Step 1100: Loss = 6.5267, Accuracy = 0.1787
Step 1150: Loss = 7.2199, Accuracy = 0.1775
Step 1200: Loss = 7.0620, Accuracy = 0.1138
Step 1250: Loss = 7.1095, Accuracy = 0.1625
Step 1300: Loss = 6.6406, Accuracy = 0.1888
Step 1350: Loss = 6.9372, Accuracy = 0.1212
Step 1400: Loss = 6.9101, Accuracy = 0.1363
Step 1450: Loss = 6.8468, Accuracy = 0.1288
Step 1500: Loss = 6.9345, Accuracy = 0.1713
Step 1550: Loss = 6.9906, Accuracy = 0.1688
Step 1600: Loss = 6.2906, Accuracy = 0.1775
Step 1650: Loss = 5.7182, Accuracy = 0.3150
Step 1700: Loss = 7.0590, Accuracy = 0.1825
Epoch 2/5 - Avg training loss: 6.7383, Accuracy: 0.1847
Validation - Loss: 6.7335, Accuracy: 0.6290, F1: 0.4985
New best model saved with F1: 0.4985
Step 1750: Loss = 6.4804, Accuracy = 0.2760
Step 1800: Loss = 6.6409, Accuracy = 0.1525
Step 1850: Loss = 6.8686, Accuracy = 0.1300
Step 1900: Loss = 6.9131, Accuracy = 0.1425
Step 1950: Loss = 6.5050, Accuracy = 0.1688
Step 2000: Loss = 6.7221, Accuracy = 0.2263
Step 2050: Loss = 7.2461, Accuracy = 0.2050
Step 2100: Loss = 6.4877, Accuracy = 0.1375
Step 2150: Loss = 7.1415, Accuracy = 0.1525
Step 2200: Loss = 6.6108, Accuracy = 0.1875
Step 2250: Loss = 6.7523, Accuracy = 0.3375
Step 2300: Loss = 6.8642, Accuracy = 0.2125
Step 2350: Loss = 6.4042, Accuracy = 0.2062
Step 2400: Loss = 6.6853, Accuracy = 0.2050
Step 2450: Loss = 6.1614, Accuracy = 0.3950
Step 2500: Loss = 6.4867, Accuracy = 0.2938
Step 2550: Loss = 6.2164, Accuracy = 0.2775
Epoch 3/5 - Avg training loss: 6.6731, Accuracy: 0.2217
Validation - Loss: 6.8134, Accuracy: 0.2833, F1: 0.2325
Early stopping triggered after 3 epochs
Training complete.
Test Accuracy: 0.6340
Test F1-Score: 0.5037
                  precision    recall  f1-score   support

         asshole       0.44      0.02      0.04       837
  everyone sucks       0.10      0.01      0.02       216
no assholes here       0.00      0.00      0.00       379
 not the asshole       0.64      0.99      0.78      2513

        accuracy                           0.63      3945
       macro avg       0.30      0.26      0.21      3945
    weighted avg       0.51      0.63      0.50      3945

Final model saved to final_aita_model.pth
Metrics saved to model_metrics.json

simplified:

Class weights: [ 6.  25.  15.   1.5]
Using device: cuda
Model and optimizer initialized.
Training with 5 epochs, early stopping with patience=1.
Epoch 1/5 - Avg training loss: 6.8579, Accuracy: 0.2390
Validation - Loss: 6.8352, Accuracy: 0.2119, F1: 0.0741
New best model saved with F1: 0.0741
Unfreezing encoder parameters...
Epoch 2/5 - Avg training loss: 6.7383, Accuracy: 0.1847
Validation - Loss: 6.7335, Accuracy: 0.6290, F1: 0.4985
New best model saved with F1: 0.4985
Epoch 3/5 - Avg training loss: 6.6731, Accuracy: 0.2217
Validation - Loss: 6.8134, Accuracy: 0.2833, F1: 0.2325
Early stopping triggered after 3 epochs
Training complete.

Test Accuracy: 0.6340
Test F1-Score: 0.5037
                  precision    recall  f1-score   support

         asshole       0.44      0.02      0.04       837
  everyone sucks       0.10      0.01      0.02       216
no assholes here       0.00      0.00      0.00       379
 not the asshole       0.64      0.99      0.78      2513

        accuracy                           0.63      3945
       macro avg       0.30      0.26      0.21      3945
    weighted avg       0.51      0.63      0.50      3945