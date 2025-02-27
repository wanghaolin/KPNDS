'''
-X_train_embed: 128-dimensional embedding of the patient -Train
-X_val_embed: 128-dimensional embedding of the patient - Valid
-X_test_embed: 128-dimensional embedding of the patient - Test
-final_classifier_pool: Base classifier pool
-threshold: Only when the bias of the base classifier is greater than the threshold value, the classifier is selected for prediction
'''

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# The MLP model is trained to select a base classifier
input_size = X_train_embed.shape[1]
hidden_size1 = 256
hidden_size2 = 128
output_size = len(final_classifier_pool)
mlp_selector = MLP(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=output_size)
# Move the MLP model to the device
mlp_selector.to(device)
criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss function
optimizer = optim.Adam(mlp_selector.parameters(), lr=0.001)
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Early stop parameter
patience = 20
best_val_loss = float('inf')
patience_counter = 0

# 100 training rounds
for epoch in range(100):
    mlp_selector.train()
    outputs = mlp_selector(X_train_embed_torch)
    loss = criterion(outputs, y_classifier_train_torch)
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    mlp_selector.eval()
    with torch.no_grad():
        val_outputs = mlp_selector(X_val_embed_torch)
        val_loss = criterion(val_outputs, y_classifier_val_torch)
    scheduler.step(val_loss)
    # Early stop
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # save model
        torch.save(mlp_selector.state_dict(), 'best_mlp_model')
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

X_test_embed_torch = torch.tensor(X_test_embed, dtype=torch.float32).to(device)
threshold = 0.8

mlp_selector.eval()
with torch.no_grad():
    classifier_probs = mlp_selector(X_test_embed_torch)
classifier_probs = torch.sigmoid(classifier_probs)
print("Classifier probabilities distribution:", classifier_probs)
# A vote prediction is made for each sample based on multiple classifiers selected by the MLP
y_pred_votes = []
y_pred_proba = []
for i in range(len(X_test)):
    sample = X_test[i].reshape(1, -1)
    sample_probs = classifier_probs[i]
    selected_classifiers_for_sample = (sample_probs > threshold).nonzero(as_tuple=False).squeeze()
    if selected_classifiers_for_sample.numel() == 0:
        selected_classifiers_for_sample = torch.argmax(sample_probs).unsqueeze(0)
    if isinstance(selected_classifiers_for_sample, int):
        selected_classifiers_for_sample = [selected_classifiers_for_sample]
    elif isinstance(selected_classifiers_for_sample, torch.Tensor):
        if selected_classifiers_for_sample.dim() == 0:
            selected_classifiers_for_sample = [selected_classifiers_for_sample.item()]
        else:
            selected_classifiers_for_sample = selected_classifiers_for_sample.tolist()

    selected_classifiers_info = [(classifier_idx, classifier_sources[classifier_idx])
                                 for classifier_idx in selected_classifiers_for_sample]
    print(f"Sample {i} selected classifiers: {selected_classifiers_info}")
    votes = []
    proba = []
    for classifier_idx in selected_classifiers_for_sample:
        classifier_with_threshold = final_classifier_pool[classifier_idx]
        probabilities = classifier_with_threshold.predict_proba(sample.reshape(1, -1))[:, 1]
        pred = classifier_with_threshold.predict(sample.reshape(1, -1))
        votes.append(pred[0])
        proba.append(probabilities.ravel())
    votes = np.array(votes).ravel()
    # Vote
    final_prediction = np.bincount(votes).argmax()
    y_pred_votes.append(final_prediction)
    if proba:
        avg_prob = np.mean(proba, axis=0)
        y_pred_proba.append(avg_prob)
y_pred = np.array(y_pred_votes)
y_pred_proba = np.array(y_pred_proba)