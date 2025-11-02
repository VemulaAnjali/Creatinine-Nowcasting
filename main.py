from utils import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier


def main():
    
    print("Loading Data...")
    
    patients = load_data("data/PATIENTS.csv")
    admissions = load_data("data/ADMISSIONS.csv")
    icustays = load_data("data/ICUSTAYS.csv")
    labevents = load_data("data/LABEVENTS.csv")
    chartevents = load_data("data/CHARTEVENTS.csv")
    d_items = load_data("data/D_ITEMS.csv")
    d_labs = load_data("data/D_LABITEMS.csv")
    
    print("Fetching Demographic Data...")
    demo_df = fetch_demographic(patients, admissions, icustays)
    
    print("Fetching Creatinine Lab and Vital Events...")
    creat_labevents, lab_mapper = fetch_labs(d_labs, labevents[labevents['hadm_id'].isin(demo_df['hadm_id'].tolist())])
    creat_chartevents, chart_mapper = fetch_chartevents(d_items, chartevents[chartevents['hadm_id'].isin(demo_df['hadm_id'].tolist())])
    
    print("Preprocessing Events")
    creat_labevents, creat_chartevents = preprocess_events(creat_labevents, creat_chartevents)
    
    print("Fetching Nowcasting Lab and Vital Data...")
    lab_dict = get_nowcastinglabs(creat_labevents, demo_df, lab_mapper)
    vital_dict = get_nowcastingvitals(creat_chartevents, demo_df, chart_mapper)
    
    print("Combining All Data...")
    full_data = get_nowcastingdata(lab_dict, vital_dict, demo_df)
    
    print("Processing Data...")
    X, y = process_data(full_data, lab_mapper)
    
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    print("===============================")
    print("Training XGBClassifier...")
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        subsample = 0.85,
        colsample_bytree = 0.9,
        n_jobs = 6,
        learning_rate = 0.1,
        eta=0.1, 
        max_depth=6, 
        seed=42 ,
        num_class=3)
    
    model.fit(X_train, y_train)
    
    print("Evaluating XGBClassifier...")
    print("-------------------------------")
    print("Metrics...")
    
    evaluate_model(model, X_test, y_test)
    
    print("===============================")
    print("===============================")
    
    print("Neural Network...")
    print("-------------------------------")
    print("Getting Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(X_train, y_train, X_test, y_test)
    
    print("Building Model...")
    nn_model = NN()
    
    print("Training Model...")
    train_acc_l, val_acc_l, train_loss_l, val_loss_l = train_nn(nn_model, train_loader, val_loader, epochs=20)
    
    print("-------------------------------")
    print("Test Performance:")
    inference(nn_model, test_loader)
    
    print("-------------------------------")
    print("Plotting Performance Curves...")
    
    plot_curves(train_acc=train_acc_l, valid_acc=val_acc_l, train_loss=train_loss_l, valid_loss=val_loss_l, epochs=len(train_acc_l))
    
    print("===============================")
    print("===============================")
    print("Feature Importance for XGBClassifier...")
    print("-------------------------------")
    shap_results(model, X_test, {**lab_mapper, **chart_mapper})
    
    print("===============================")
    print("===============================")
    print("Ablation Study...")
    
    ablation_1, ablation_2 = ablation_study(X)
    
    print("-------------------------------")
    print("1) Dropping Creatinine's Abnormal Flag...")
    
    
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(ablation_1, y, test_size=0.33, random_state=42)
    model1 = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        subsample = 0.85,
        colsample_bytree = 0.9,
        n_jobs = 6,
        learning_rate = 0.1,
        eta=0.1, 
        max_depth=6, 
        seed=42 ,
        num_class=3)
    
    model1.fit(X_train1, y_train1)
    print("Metrics...")
    evaluate_model(model1, X_test1, y_test1)
    
    print("-------------------------------")
    print("Feature Importance...")
    
    shap_results(model1, X_test1, {**lab_mapper, **chart_mapper})
    
    print("-------------------------------")
    print("2) Dropping Time-specific Features...")
    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(ablation_2, y, test_size=0.33, random_state=42)
    model2 = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        subsample = 0.85,
        colsample_bytree = 0.9,
        n_jobs = 6,
        learning_rate = 0.1,
        eta=0.1, 
        max_depth=6, 
        seed=42 ,
        num_class=3)
    
    model2.fit(X_train2, y_train2)
    print("Metrics...")
    evaluate_model(model2, X_test2, y_test2)
    
    print("-------------------------------")
    print("Feature Importance...")
    
    shap_results(model2, X_test2, {**lab_mapper, **chart_mapper})
    
    print("-------------------------------")
    print("Done.")
    

if __name__ == "__main__": 
    main()