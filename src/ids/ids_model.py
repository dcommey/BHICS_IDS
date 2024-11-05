import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling1D, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def create_cnn_model(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_lstm_model(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)  # Added BatchNorm
    x = Dropout(0.3)(x)
    x = LSTM(32, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)  # Added BatchNorm
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)  # Added BatchNorm
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_random_forest_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def create_xgboost_model():
    return xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric=['error', 'logloss'],
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=10
    )

def lr_schedule(epoch):
    initial_lr = 1e-3
    if epoch > 10:
        return initial_lr * 0.1
    elif epoch > 20:
        return initial_lr * 0.01
    return initial_lr

def compile_and_fit(model, X_train, y_train, X_val, y_val, class_weight=None, epochs=100, batch_size=128):
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight,
        verbose=1
    )
    
    return history