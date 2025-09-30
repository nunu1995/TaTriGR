# Global constants (last criterion must be 'overall')

# RB
CRITERIA = ['Appearance', 'Aroma', 'Palate', 'Taste', 'overall']
# YM
# CRITERIA = ['Visuals', 'Direction', 'Story', 'Acting', 'overall']
# TA
# CRITERIA = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Checkin', 'Business', 'overall']

DEFAULTS = {
    'embedding_size': 64,
    'n_layers': 2,
    'reg_weight': 1e-4,
    'uc_alpha': 2.0,
    'ic_beta': 10.0,
    'aux_ipd_weight': 1e-3,
    'aux_lexi_weight': 1e-3,
    'epochs': 150,
    'train_batch_size': 2048,
    'eval_batch_size': 4096,
    'metrics': ['Hit', 'Recall', 'NDCG'],
    'topk': [20, 50],
    'seed': 42,
}