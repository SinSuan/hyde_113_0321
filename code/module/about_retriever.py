"""123
"""

import numpy as np
import faiss
from pyserini.search.lucene import LuceneSearcher
from FlagEmbedding import BGEM3FlagModel

def init_retriever(retriever_type, path_2_corpus):
    """123
    """
    print("enter init_retriever")

    if retriever_type=='BM25':
        print("\tenter if 'BM25'")

        # init BM25
        # connect BM25 to corpus
        retriever = LuceneSearcher(path_2_corpus)
        retriever.set_language('zh')

        print("\texit if 'BM25'")

    elif retriever_type=='BGEM3':
        print("\tenter elif 'BGEM3'")

        # init faiss
        d = 1024
        retriever = faiss.IndexFlatL2(d)

        # connect faiss to corpus
        corpus = np.load(path_2_corpus)
        retriever.add(corpus)   # 這行沒有 bug，跟官方文件寫法一樣，且跑起來是正確的

        print("\texit elif 'BGEM3'")

    retriever_info = [retriever_type, retriever]

    print("exit init_retriever")
    return retriever_info

def apply_retriever(retriever_info, ttl_query, top_k=15):
    r"""

    Parameters
    ----------
    query : List[str]
        total queries

    Returns
    -------
    List[List[int]]
        retrieved docid(s) for each query
    """
    print("enter apply_retriever")

    retriever_type, retriever = retriever_info

    if retriever_type=='BM25':
        print("\tenter if 'BM25'")

        ttl_hits_id = []
        for query in ttl_query:
            hits = retriever.search(query, top_k)
            hits = [ hit.docid for hit in hits ]
            ttl_hits_id.append(hits)

        print("\texit if 'BM25'")

    elif retriever_type=='BGEM3':
        print("\tenter elif 'BGEM3'")

        encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        query_vec = encoder.encode(
            ttl_query,
            batch_size=12,
            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
            )['dense_vecs']

        ttl_hits_id = retriever.search(query_vec, top_k)

        print("\texit elif 'BGEM3'")

    print("exit apply_retriever")
    return ttl_hits_id

# if __name__ == "__main__":
#     # retriever_type = 'BM25'
#     # path_2_ttl_data = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/raw_data'
#     # path_2_corpus = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/BM25'

#     retriever_type = 'BGEM3'
#     path_2_ttl_data = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/raw_data/raw_data.json'
#     path_2_corpus = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/BGEM3/113_0322_1917.npy'

#     ttl_query = ['松鼠吃葡萄嗎?']
#     top_k = 15
    
#     corpus_build(retriever_type, path_2_ttl_data, path_2_corpus)
#     retriever_info = init_retriever(retriever_type, path_2_corpus)
#     ttl_hits = apply_retriever(retriever_info, ttl_query, top_k=15)
    
#     print(f"ttl_hits = {ttl_hits}")