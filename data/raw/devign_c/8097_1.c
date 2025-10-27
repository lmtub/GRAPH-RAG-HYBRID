static inline int get_chroma_qp(int chroma_qp_index_offset, int qscale){



    return chroma_qp[av_clip(qscale + chroma_qp_index_offset, 0, 51)];

}
