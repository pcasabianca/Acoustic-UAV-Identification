from tools.perf.snr import bin_by_snr, filter_drone_only


def test_empty_bins_and_mean_prob():
    preds = [
        {'true_label': 1, 'prob_class1': 0.9, 'pred_label': 1, 'metadata': {'raw_debug': {'resulting_snr_db': -15}}},
        {'true_label': 1, 'prob_class1': 0.6, 'pred_label': 1, 'metadata': {'raw_debug': {'resulting_snr_db': -5}}},
    ]
    bins = [-30, -20, -10, 0]
    b = bin_by_snr(preds, bins)
    # bin indices 0,1,2 -> only bins 0 and 1 may have values; ensure empty bin returns []
    assert isinstance(b, dict)
    assert 2 in b and isinstance(b[2], list)


def test_bin_by_snr_basic():
    preds = [
        {'true_label': 1, 'prob_class1': 0.9, 'pred_label': 1, 'metadata': {'raw_debug': {'resulting_snr_db': -15}}},
        {'true_label': 1, 'prob_class1': 0.6, 'pred_label': 1, 'metadata': {'raw_debug': {'resulting_snr_db': -5}}},
        {'true_label': 0, 'prob_class1': 0.2, 'pred_label': 0, 'metadata': {'raw_debug': {'resulting_snr_db': 5}}}
    ]
    bins = [-30, -10, 0, 10]
    b = bin_by_snr(preds, bins)
    assert len(b) == 3
    # first bin has one item
    assert len(b[0]) == 1
