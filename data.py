import os
import pandas as pd


def load_from_inter_files(dataset_name: str, criteria: list, inter_dir: str = '.',
                          train_split: str = 'tr', valid_split: str = 'val', test_split: str = 'ts'):
    """Read *.inter files and return:
       - df_train_plain: [user_id, item_id, <sub-criteria>, overall] with clean column names
       - n_users, n_items, n_criteria
    """
    def _read(split):
        path = os.path.join(inter_dir, f'{dataset_name}/{dataset_name}.{split}.inter')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Not found: {path}')
        df = pd.read_csv(path, sep='\t')
        # normalize RecBole typed headers
        rename = {}
        if 'user_id' in df.columns: rename['user_id'] = 'user_id:token'
        if 'item_id' in df.columns: rename['item_id'] = 'item_id:token'
        for c in criteria:
            if c in df.columns: rename[c] = f'{c}:float'
        if 'rating' in df.columns: rename['rating'] = 'rating:float'
        df.rename(columns=rename, inplace=True)
        return df

    df_tr = _read(train_split)
    df_val = _read(valid_split)
    df_ts = _read(test_split)

    ucol = 'user_id:token'
    icol = 'item_id:token'
    sub_criteria = criteria[:-1]
    sub_cols = [f'{c}:float' for c in sub_criteria]
    overall_col = f'{criteria[-1]}:float'

    n_users = int(max(df_tr[ucol].max(), df_val[ucol].max(), df_ts[ucol].max()) + 1)
    n_items = int(max(df_tr[icol].max(), df_val[icol].max(), df_ts[icol].max()) + 1)
    n_cri = len(criteria)

    df_train_plain = pd.DataFrame({
        'user_id': df_tr[ucol].astype(int),
        'item_id': df_tr[icol].astype(int)
    })
    for c in sub_criteria:
        df_train_plain[c] = df_tr[f'{c}:float'].astype(float)
    df_train_plain[criteria[-1]] = df_tr[overall_col].astype(float)

    return df_train_plain, n_users, n_items, n_cri


def collect_ui_pairs_from_recbole(train_data, uid_field: str, iid_field: str, n_users: int, n_items: int):
    """Collect observed (u,i) from RecBole internal IDs in training loader."""
    pairs = set()
    for batch in train_data:
        u = batch[uid_field].cpu().numpy()
        i = batch[iid_field].cpu().numpy()
        for uu, ii in zip(u, i):
            if 0 <= uu < n_users and 0 <= ii < n_items:
                pairs.add((int(uu), int(ii)))
    if not pairs:
        raise RuntimeError("No (u,i) pairs collected from RecBole training data.")
    return pairs
