import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import rasterio

def load_data_and_labels(data_paths, label_paths):
    all_data = []
    all_labels = []
    original_shapes = []
    valid_indices_list = []
    
    for data_path, label_path in zip(data_paths, label_paths):
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Data or label file not found for paths: {data_path}, {label_path}")
        
        data = np.load(data_path)  # Shape: (time, channel, height, width)
        labels = np.load(label_path)  # Shape: (height, width)
        original_shapes.append(labels.shape)
        
        labels = np.where(labels > 3, 0, labels)
        labels = np.where(labels == 0, np.nan, labels)
        
        labels_expanded = np.repeat(labels[np.newaxis, :, :], data.shape[0], axis=0)
        
        # Calculate statistics for each pixel
        data_min = np.nanmin(data[:,6:,:,:], axis=0)
        data_max = np.nanmax(data[:,6:,:,:], axis=0)
        #data_median = np.nanmedian(data[6:], axis=0)
        data_std = np.nanstd(data[:,6:,:,:], axis=0)
        #data_75th = np.nanpercentile(data[6:], 75, axis=0)
        data_avg = np.nanmean(data[:,6:,:,:], axis=0)
        
        # Stack the statistics along the channel dimension
        data_stats = np.concatenate([data_min,data_max, data_std, data_avg], axis=0)

        # Reshape to (height * width, num_features)
        data_reshaped = data_stats.reshape(-1, data_stats.shape[1] * data_stats.shape[2])

        labels_reshaped = labels_expanded[0].ravel()  # Flatten labels

        valid_indices = ~np.isnan(data_reshaped).any(axis=0) & ~np.isnan(labels_reshaped)
        data_reshaped_clean = data_reshaped[:,valid_indices]
        labels_reshaped_clean = labels_reshaped[valid_indices]

        all_data.append(data_reshaped_clean)
        all_labels.append(labels_reshaped_clean)
        valid_indices_list.append(valid_indices)
    
    return np.hstack(all_data), np.concatenate(all_labels), original_shapes, valid_indices_list

def plot_confusion_matrix(y_true, y_pred, labels, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, cbar=False)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.5, f'{cm[i,j]}\n{cm_percent[i,j]:.2f}%',
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)

def print_metrics(y_true, y_pred, set_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    user_accuracy = (y_true == y_pred).sum() / len(y_true) * 100  # User accuracy as a percentage
    
    print(f"\n{set_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"User Accuracy: {user_accuracy:.2f}%")  # Display user accuracy

def plot_test_map(y_true, y_pred, original_shapes, valid_indices, model_name, label_tiff_path):
    cmap = ListedColormap(['#ffa500', '#006400', '#800080'])
    
    shape = original_shapes
    y_pred_map = np.full(shape, np.nan)
    y_pred_map.flat[valid_indices] = y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    im = ax.imshow(y_pred_map, cmap=cmap, vmin=0, vmax=2, aspect='equal')
    ax.set_title('Predicted Labels', fontsize=14)
    fig.colorbar(im, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{model_name}_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    with rasterio.open(label_tiff_path) as src:
        profile = src.profile
        transform = src.transform

    pred_tiff_path = f'{model_name}_test.tiff'
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    with rasterio.open(pred_tiff_path, 'w', **profile) as dst:
        dst.write(y_pred_map.astype(rasterio.float32), 1)
        dst.transform = transform

def plot_feature_importance(model, X_train, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importances for {model_name}')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), np.array(feature_names)[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_model(model, X_train, y_train, X_test, y_test, test_shapes, test_valid_indices, test_label_tiff_path, model_name):
    print(f"Evaluating {model_name}...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_user_accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f'Fold {fold + 1}/5')
        
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        
        y_val_pred = model.predict(X_val_fold)
        
        fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_precision = precision_score(y_val_fold, y_val_pred, average='weighted')
        fold_recall = recall_score(y_val_fold, y_val_pred, average='weighted')
        fold_f1 = f1_score(y_val_fold, y_val_pred, average='weighted')
        fold_user_accuracy = (y_val_fold == y_val_pred).sum() / len(y_val_fold) * 100
        
        fold_accuracies.append(fold_accuracy)
        fold_precisions.append(fold_precision)
        fold_recalls.append(fold_recall)
        fold_f1s.append(fold_f1)
        fold_user_accuracies.append(fold_user_accuracy)
        
        print_metrics(y_val_fold, y_val_pred, f"Validation Fold {fold + 1}")

    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)
    avg_f1 = np.mean(fold_f1s)
    avg_user_accuracy = np.mean(fold_user_accuracies)

    print(f"\nAverage Validation Metrics over {kf.get_n_splits()} folds for {model_name}:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"User Accuracy: {avg_user_accuracy:.2f}%")

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    plot_confusion_matrix(y_train, model.predict(X_train), 
                          labels=['other_crops', 'coffee', 'intercropped_coffee'],
                          title=f'{model_name} Training Confusion Matrix', ax=ax1)

    plot_confusion_matrix(y_test, y_test_pred, 
                          labels=['other_crops', 'coffee', 'intercropped_coffee'],
                          title=f'{model_name} Testing Confusion Matrix', ax=ax2)

    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    print_metrics(y_train, model.predict(X_train), f"{model_name} Training")
    print_metrics(y_test, y_test_pred, f"{model_name} Testing")
    plot_feature_importance(model, X_train, model_name)
    plot_test_map(y_test, y_test_pred, test_shapes, test_valid_indices, model_name, test_label_tiff_path)

    print(f"All plots for {model_name} have been saved.")

train_data_paths = ['data_G_train.npy','data_D_train.npy']
train_label_paths = ['labels_G_train.npy','labels_D_train.npy']
test_data_paths = ['data_G_test.npy']
test_label_paths = ['labels_G_test.npy']
test_label_tiff_paths = ['labels_G_tiff.tif','labels_D_tiff.tif']

X_train, y_train, train_shapes, train_valid_indices = load_data_and_labels(train_data_paths, train_label_paths)

X_test_list, y_test_list, test_shapes_list, test_valid_indices_list = [], [], [], []
for data_path, label_path in zip(test_data_paths, test_label_paths):
    X_test, y_test, test_shapes, test_valid_indices = load_data_and_labels([data_path], [label_path])
    X_test_list.append(X_test)
    y_test_list.append(y_test)
    test_shapes_list.append(test_shapes[0])
    test_valid_indices_list.append(test_valid_indices[0])

y_train -= 1
y_test_list = [y_test - 1 for y_test in y_test_list]

X_train = X_train.transpose()
X_test_list = [X_test.transpose() for X_test in X_test_list]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_list = [scaler.transform(X_test) for X_test in X_test_list]

rf_params = {'bootstrap': True, 'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1}
rf_model = RandomForestClassifier(**rf_params, random_state=42)
xgb_params = {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.2, 'subsample': 1.0}
xgb_model = xgb.XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss', random_state=42)

for i, (X_test, y_test, test_shapes, test_valid_indices) in enumerate(zip(X_test_list, y_test_list, test_shapes_list, test_valid_indices_list)):
    print(f"Evaluating on test dataset {i + 1}...")
    evaluate_model(rf_model, X_train, y_train, X_test, y_test, test_shapes, test_valid_indices, test_label_tiff_paths[i], f"RandomForest_Test_{i + 1}")
    evaluate_model(xgb_model, X_train, y_train, X_test, y_test, test_shapes, test_valid_indices, test_label_tiff_paths[i], f"XGBoost_Test_{i + 1}")
