"""123
"""
import os
import configparser
from dotenv import load_dotenv

from component import hyde_total, corpus_build, retrieve_related_docid, save_logging
from module.time_now import time_now

QUERY_DOCID = {
    "我的茶樹周遭出現小土推跟蟲孔，請問發生什麼事了？" : "147937",
    "甘藍葉片出現黃黑小蟲，應該噴灑什麼藥劑？" : "147934",
    "果樹葉片上出現白粉" : "147870",
    "果樹出現煤煙病菌" : "147870",
    "我的番茄樹出現不明卵塊" : "147871"
}

TTL_DOCID = QUERY_DOCID.values()
TTL_DOCID = [int(idx) for idx in set(TTL_DOCID)]
TTL_QUERY = list(QUERY_DOCID.keys())


def main():
    """123
    """
    
    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    retriever_type = config['mode']['retriever_type']
    now = time_now()

    # hyde corpus
    path_2_raw_data = config['data']['raw_data']
    _, path_2_hyde_data = hyde_total(now, path_2_raw_data, TTL_DOCID)

    # create corpus
    dir_2_corpus = os.path.join(config['corpus_dir'][retriever_type], now)
    os.makedirs(dir_2_corpus, exist_ok=True)
    if retriever_type=='BGEM3' :
        dir_2_corpus += f"/{now}.npy"
    elif retriever_type=='BM25' :
        path_2_hyde_data = os.path.dirname(path_2_hyde_data)
    corpus_build(path_2_hyde_data, dir_2_corpus)

    # retrieve
    ttl_hints_id = retrieve_related_docid(dir_2_corpus, TTL_QUERY, 15)

    # sava log path
    save_logging(now, dir_2_corpus, QUERY_DOCID, ttl_hints_id)

    print(f"ttl_hints_id = {ttl_hints_id}")


if __name__ == "__main__":
    load_dotenv("/user_data/DG/hyde_113_0321/global_variable/.env")
    main()
