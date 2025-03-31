import lightgbm as lgb

def train_gbm(X_train, y_train, X_test, y_test, params=None, num_boost_round=100):
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    if params:
        default_params.update(params)
    
    gbm_model = lgb.train(
        default_params, 
        lgb_train, 
        num_boost_round=num_boost_round, 
        valid_sets=[lgb_train, lgb_eval], 
        valid_names=['train', 'eval']
    )
    
    return gbm_model

def predict_gbm(gbm_model, X_test):
    y_pred_gbm = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration)
    return y_pred_gbm
