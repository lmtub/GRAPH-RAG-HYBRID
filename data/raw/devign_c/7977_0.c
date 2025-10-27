static inline int get_chroma_qp(H264Context *h, int t, int qscale){

    return h->pps.chroma_qp_table[t][qscale];

}
