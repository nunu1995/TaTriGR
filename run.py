import time
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from tatriGR.config import CRITERIA, DEFAULTS
from tatriGR.data import load_from_inter_files, collect_ui_pairs_from_recbole
from tatriGR.graph import build_tripartite_adj_and_weights
from tatriGR.model import TaTriGR
from tatriGR.utils import count_parameters


def main():
    DATASET_NAME = 'RB'
    INTER_DIR = '.'

    # Load *.inter to build graph
    df_train_plain, n_users, n_items, n_cri = load_from_inter_files(
        dataset_name=DATASET_NAME, criteria=CRITERIA, inter_dir=INTER_DIR,
        train_split='tr', valid_split='val', test_split='ts'
    )

    # Recbole config
    parameter_dict = {
        'model': 'TaTriGR',
        'dataset': DATASET_NAME,
        'data_path': INTER_DIR,
        'benchmark_filename': ['tr', 'val', 'ts'],
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',

        'neg_sampling': {'uniform': 1},
        'epochs': DEFAULTS['epochs'],
        'metrics': DEFAULTS['metrics'],
        'topk': DEFAULTS['topk'],
        'train_batch_size': DEFAULTS['train_batch_size'],
        'eval_batch_size': DEFAULTS['eval_batch_size'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Model
        'embedding_size': DEFAULTS['embedding_size'],
        'n_layers': DEFAULTS['n_layers'],
        'reg_weight': DEFAULTS['reg_weight'],

        # Alpha / Beta
        'uc_alpha': DEFAULTS['uc_alpha'],
        'ic_beta': DEFAULTS['ic_beta'],

        # Aux losses
        'aux_ipd_weight': DEFAULTS['aux_ipd_weight'],
        'aux_lexi_weight': DEFAULTS['aux_lexi_weight'],

        'seed': DEFAULTS['seed'], 'reproducibility': True,
    }

    config = Config(model=TaTriGR, dataset=DATASET_NAME, config_dict=parameter_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Instantiate model
    model = TaTriGR(config=config, dataset=train_data.dataset, n_cri=len(CRITERIA)).to(config['device'])

    # Collect observed (u,i) pairs from RecBole internal IDs
    uid_field = model.USER_ID
    iid_field = model.ITEM_ID
    ui_pairs_internal = collect_ui_pairs_from_recbole(
        train_data, uid_field, iid_field, model.n_users, model.n_items
    )

    # Build graph + weights per paper
    adj, w_uc, tilde_ic = build_tripartite_adj_and_weights(
        ui_pairs_internal=ui_pairs_internal,
        df_train_plain=df_train_plain,
        n_users=model.n_users, n_items=model.n_items, criteria=CRITERIA,
        uc_alpha=float(config.get('uc_alpha', 2.0)),
        ic_beta=float(config.get('ic_beta', 10.0)),
        lambda_ui=1.0, lambda_uc=1.0, lambda_ic=1.0
    )
    model.encoder.adj = adj.to(config['device'])
    model.user_w_uc = w_uc.to(config['device']).float()
    model.item_tilde_ic = tilde_ic.to(config['device']).float()

    # Train & evaluate
    trainer = Trainer(config, model)
    t0 = time.time()
    test_result = trainer.evaluate(test_data)
    logger.info(f'[Test] {test_result}')
    print(test_result)
    print(f'Inference time: {time.time()-t0:.3f}s')


if __name__ == '__main__':
    main()
