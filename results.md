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

