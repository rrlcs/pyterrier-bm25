import pyterrier as pt
import pandas as pd


def get_dataframe():
    pdes = pd.read_csv("data/product_descriptions.csv",
                       on_bad_lines="skip", engine="python")

    train = pd.read_csv("data/train.csv", on_bad_lines="skip",
                        encoding="unicode_escape", engine="python")

    att = pd.read_csv('data/attributes.csv')

    return pdes, train, att


def preprocess(train, pdes, att):
    # Convert relevance scores into binary
    train['relevance'] = (train['relevance'] > 2).astype(int)

    # Add column query id (qid)
    train['qid'] = train.groupby('search_term').ngroup()

    train_with_pdes = (
        pd.merge(train, pdes, on="product_uid", how="left")).drop_duplicates()

    att = att[['product_uid', 'value']]
    att = att.groupby('product_uid').agg(lambda x: x.tolist())
    att['value'] = att['value'].str.join('#')

    train_product_info = (pd.merge(train_with_pdes, att,
                          on='product_uid', how='left')).fillna('')

    train_product_info['product_info'] = ((train_product_info['product_title']) + "$" +
                                          (train_product_info['product_description']) +
                                          "$" + (train_product_info['value'])).replace('\n', '')

    train_product_info.to_csv('df_preprocessed.csv')

    return train_product_info


def train_test_split(train_product_info):
    df_train = train_product_info.sample(frac=0.01, random_state=1)
    df_train['product_uid'] = df_train['product_uid'].apply(lambda x: str(x))
    df_train['product_info'] = df_train['product_info'].apply(lambda x: str(x))

    df_test = train_product_info.drop(df_train.index)

    return df_train, df_test


def init_pyterrier():
    if not pt.started():
        pt.init()


def create_index(df_train_formatted):
    pd_indexer = pt.DFIndexer("./pd_index")
    indexref = pd_indexer.index(
        df_train_formatted["text"], df_train_formatted["docno"])

    return indexref


def print_stats(indexref):
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())


def search_query(indexref, q):
    print(pt.BatchRetrieve(indexref).search(q))


def get_query(df_train):
    df_query = df_train[['qid', 'search_term']]
    df_query = df_query.rename(columns={'search_term': 'query'})
    query = df_query['query'].to_list()
    query = [q.replace('/', '') for q in query]
    query = [q.replace('\'', '') for q in query]
    df_query['query'] = query

    return df_query


def retrieve_results(df_query, indexref):
    retr = pt.BatchRetrieve(indexref, wmodel="BM25")
    res = retr(df_query)

    return res


def get_qrels_data(df_train):
    qrels = df_train[['qid', 'product_uid', 'relevance']]
    qrels = qrels.assign(iter=0)
    qrels = qrels[['qid', 'iter', 'product_uid', 'relevance']]
    qrels = qrels.rename(columns={'product_uid': 'doc_id'})
    qrels = qrels.astype(str)
    qrels['relevance'] = qrels['relevance'].astype(int)

    return qrels


def evaluate(res, qrels):
    eval = pt.Utils.evaluate(res, qrels)

    return eval


if __name__ == "__main__":
    pdes, train, att = get_dataframe()

    train_product_info = preprocess(train, pdes, att)

    df_train, df_test = train_test_split(train_product_info)

    uid = df_train['product_uid'].to_list()
    text = df_train['product_info'].to_list()

    df_train_formatted = pd.DataFrame({
        'docno':
        uid,
        'text':
        text
    })

    init_pyterrier()

    indexref = create_index(df_train_formatted)
    print_stats(indexref)

    search_query(indexref, q="angle bracket")

    df_query = get_query(df_train)

    res = retrieve_results(df_query, indexref)

    qrels = get_qrels_data(df_train)

    # Evaluate
    eval = evaluate(res, qrels)
    print("eval: ", eval)
