from vn1.data import load_full_data, process_wide_df, trim_leading_zeros
from vn1.ensemble import apply_weights, optimize_weights
from vn1.models import fit_predict_lgbm, fit_predict_stats
from vn1.submission import apply_zero_variance_override, write_submission
from vn1.validation import cross_validate, evaluate_predictions

if __name__ == "__main__":
    sales = trim_leading_zeros(process_wide_df(load_full_data())).collect()

    cv_stats = cross_validate(lambda tr, h: fit_predict_stats(tr, h=h), sales, h=13, n_windows=4)
    cv_lgbm = cross_validate(lambda tr, h: fit_predict_lgbm(tr, h=h), sales, h=13, n_windows=4)
    cv = cv_stats.join(cv_lgbm.select("unique_id", "ds", "fold", "LGBM"), on=["unique_id", "ds", "fold"])

    print(evaluate_predictions(cv, pred_cols=["Theta", "AutoETS", "SNaive", "LGBM"]))

    weights = optimize_weights(cv, pred_cols=["Theta", "AutoETS", "SNaive", "LGBM"])
    print("ensemble weights:", weights)

    # refit on full data, blend, override, write
    stats_full = fit_predict_stats(sales, h=13)
    lgbm_full = fit_predict_lgbm(sales, h=13, use_cv=True)
    preds = apply_weights(
        stats_full.join(lgbm_full, on=["unique_id", "ds"]), weights, out_col="y_hat"
    ).collect()
    write_submission(apply_zero_variance_override(preds, sales).collect(), "artifacts/submission.csv")
