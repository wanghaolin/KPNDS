'''
-X_train_embed: 128-dimensional embedding of the patient -Train
-X_val_embed: 128-dimensional embedding of the patient - Valid
-X_test_embed: 128-dimensional embedding of the patient - Test
-final_classifier_pool: Base classifier pool
-threshold: Only when the bias of the base classifier is greater than the threshold value, the classifier is selected for prediction
'''


input_size = X_train_embed.shape[1]
output_size = len(final_classifier_pool)
kan_selector = KAN(width=[input_size, 64, output_size], grid=1, k=3, device=device)
kan_selector.to(device)
criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss function
optimizer = optim.Adam(kan_selector.parameters(), lr=0.001, weight_decay=0.1)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
# Early stop parameter
patience = 20
best_val_loss = float('inf')
patience_counter = 0

# 100 training rounds
for epoch in range(100):
    kan_selector.train()
    outputs = kan_selector(X_train_embed_torch)
    loss = criterion(outputs, y_classifier_train_torch)
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    kan_selector.eval()
    with torch.no_grad():
        val_outputs = kan_selector(X_val_embed_torch)
        val_loss = criterion(val_outputs, y_classifier_val_torch)
    scheduler.step(val_loss)
    # Early stop
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # save model
        torch.save(kan_selector.state_dict(), 'best_kan_model')
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Select a base classifier in the test set
X_test_embed_torch = torch.tensor(X_test_embed, dtype=torch.float32).to(device)
threshold = 0.8

kan_selector.eval()
with torch.no_grad():
    classifier_probs = kan_selector(X_test_embed_torch)
classifier_probs = torch.sigmoid(classifier_probs)
print("Classifier probabilities distribution:", classifier_probs)
# Vote Prediction
y_pred_votes = []
y_pred_proba = []
selected_classifiers_matrix = np.zeros((len(X_test), len(final_classifier_pool)))
for i in range(len(X_test)):
    sample = X_test[i].reshape(1, -1)
    sample_probs = classifier_probs[i]
    selected_classifiers_matrix[i, :] = sample_probs.cpu().numpy()

    # Select a classifier with a confidence greater than the threshold
    selected_classifiers_for_sample = (sample_probs > threshold).nonzero(as_tuple=False).squeeze()

    # If no classifier has a confidence above the threshold, the classifier with the highest confidence is selected
    if selected_classifiers_for_sample.numel() == 0:
        selected_classifiers_for_sample = torch.argmax(sample_probs).unsqueeze(0)

    # Make sure that selected_classifiers_for_sample is a one-dimensional tensor
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

    # Prediction
    votes = []
    proba = []
    for classifier_idx in selected_classifiers_for_sample:
        classifier_with_threshold = final_classifier_pool[classifier_idx]
        probabilities = classifier_with_threshold.predict_proba(sample.reshape(1, -1))[:, 1]
        pred = classifier_with_threshold.predict(sample.reshape(1, -1))
        votes.append(pred[0])
        proba.append(probabilities.ravel())

    votes = np.array(votes).ravel()
    # VOte
    final_prediction = np.bincount(votes).argmax()
    y_pred_votes.append(final_prediction)
    if proba:
        avg_prob = np.mean(proba, axis=0)
        y_pred_proba.append(avg_prob)
y_pred = np.array(y_pred_votes)
y_pred_proba = np.array(y_pred_proba)