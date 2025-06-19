'''
-X_train_embed: 128-dimensional embedding of the patient -Train
-X_val_embed: 128-dimensional embedding of the patient - Valid
-X_test_embed: 128-dimensional embedding of the patient - Test
-final_classifier_pool: Base classifier pool
-threshold: Only when the bias of the base classifier is greater than the threshold value, the classifier is selected for prediction
'''

# Transformer model
class PositionalEncoding(nn.Module):
    """A positional encoding layer that adds positional information to a sequence"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerSelector(nn.Module):
    """Transformer-based metaclassifier selector"""

    def __init__(self, input_dim, output_dim, d_model=128, nhead=4,
                 num_layers=3, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        """Parameter initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # Input shape: (batch_size, input_dim)
        src = self.embedding(src)  # (batch_size, d_model)
        src = src.unsqueeze(1)  # (batch_size, 1, d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src)  # (batch_size, 1, d_model)
        output = output.squeeze(1)  # (batch_size, d_model)
        return self.fc_out(output)  # (batch_size, output_dim)

input_size = X_train_embed.shape[1]
output_size = len(final_classifier_pool)
# Initialize the Transformer selector
transformer_selector = TransformerSelector(
    input_dim=input_size,
    output_dim=output_size,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.2
).to(device)
criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss function
optimizer = optim.AdamW(transformer_selector.parameters(), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# Early stop parameter
patience = 20
best_val_loss = float('inf')
patience_counter = 0

# 100 training rounds
for epoch in range(100):
    transformer_selector.train()
    outputs = transformer_selector(X_train_embed_torch)
    loss = criterion(outputs, y_classifier_train_torch)
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    transformer_selector.eval()
    with torch.no_grad():
        val_outputs = transformer_selector(X_val_embed_torch)
        val_loss = criterion(val_outputs, y_classifier_val_torch)
    scheduler.step(val_loss)
    # Early stop
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # save model
        torch.save(transformer_selector.state_dict(), 'best_transformer_model')
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Select a base classifier in the test set
X_test_embed_torch = torch.tensor(X_test_embed, dtype=torch.float32).to(device)
threshold = 0.7

transformer_selector.eval()
with torch.no_grad():
    classifier_logits = transformer_selector(X_test_embed_torch)
    classifier_probs = torch.sigmoid(classifier_logits)
transformer_selector.eval()
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