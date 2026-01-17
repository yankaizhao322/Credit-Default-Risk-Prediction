import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def run_np_analysis(model, model_name, X_val, y_val, alpha=0.01):
    print(f"Running NP Analysis for: {model_name}")

    # 1. Split Calibration / Test

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42, stratify=y_val
    )

    # 2. Get Predictions
    try:
        if hasattr(model, "predict_proba"):
            # print(" -> Mode: Scikit-Learn API (predict_proba)")
            y_cal_proba = model.predict_proba(X_cal)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, "predict"):
            model_type_str = str(type(model))

            if "xgboost" in model_type_str:
                print(" -> Mode: Native XGBoost (DMatrix)")
                dcal = xgb.DMatrix(X_cal)
                dtest = xgb.DMatrix(X_test)
                y_cal_proba = model.predict(dcal)
                y_test_proba = model.predict(dtest)

            elif "lightgbm" in model_type_str:
                print(" -> Mode: Native LightGBM (Direct Predict)")
                y_cal_proba = model.predict(X_cal)
                y_test_proba = model.predict(X_test)

            else:
                raise AttributeError("Unsupported model type for NP analysis.")

        else:
            raise AttributeError("Model has neither predict_proba nor predict methods.")

    except Exception as e:
        print(f"\n[Error] prediction errors: {e}")
        return 0.0

    scores_cal = y_cal_proba
    labels_cal = y_cal

    # 3. Find Threshold Strategy: Iterate candidates to max recall under FPR constraint
    candidate_thresholds = np.unique(scores_cal)[::-1]
  
    best_t = None
    best_recall = -1.0
    best_fpr = None

    for t in candidate_thresholds:

        preds = (scores_cal >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels_cal, preds, labels=[0, 1]).ravel()

        if (fp + tn) == 0 or (tp + fn) == 0:
            continue

        fpr_cal = fp / (fp + tn)
        recall_cal = tp / (tp + fn)

        if fpr_cal <= alpha and recall_cal > best_recall:
            best_recall = recall_cal
            best_t = t
            best_fpr = fpr_cal

    # Fallback mechanism
    if best_t is None:

        neg_scores = scores_cal[labels_cal == 0]
        threshold = np.percentile(neg_scores, (1 - alpha) * 100)
        print(" No threshold with Cal FPR <= alpha found, fallback to percentile method.")
        cal_fpr_msg = "N/A (percentile fallback)"
        cal_recall_msg = "N/A (percentile fallback)"

    else:
        threshold = best_t
        cal_fpr_msg = f"{best_fpr:.2%}"
        cal_recall_msg = f"{best_recall:.2%}"

    print(f" Target FPR: <= {alpha:.1%}")
    print(f" Calibration FPR at threshold: {cal_fpr_msg}")
    print(f" Calibration Recall at threshold: {cal_recall_msg}")
    print(f" Calculated Threshold: {threshold:.4f}")

    # 4. Apply to Test Set

    pred_test = (y_test_proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()

    realized_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    realized_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0


    print(f"\n[Test Set Performance]")
    print(f" Realized FPR:   {realized_fpr:.2%}  ({'### Pass' if realized_fpr <= alpha + 0.015 else 'High Risk'})")
    print(f" Realized Recall: {realized_recall:.2%} (Approx. max power under compliance)")

    # 5. Bootstrap

    print(f"\n[Running Bootstrap Test (1000 iter)...]")
    boot_recalls = []
    np.random.seed(2025)

    n_test = len(y_test_proba)
  
    for _ in range(1000):
        indices = np.random.choice(n_test, n_test, replace=True)
        y_true_boot = y_test[indices]
        y_score_boot = y_test_proba[indices]

        p_boot = (y_score_boot > threshold).astype(int)

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_boot, p_boot, labels=[0, 1]).ravel()
        

        if (tp_b + fn_b) > 0:
            boot_recalls.append(tp_b / (tp_b + fn_b))
        else:
            boot_recalls.append(0.0)

    avg_recall = np.mean(boot_recalls)
    print(f" -> Bootstrap Average Recall: {avg_recall:.2%}")

    

    return avg_recall
