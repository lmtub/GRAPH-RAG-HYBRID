static av_always_inline void dist_scale(HEVCContext *s, Mv *mv,

                                        int min_pu_width, int x, int y,

                                        int elist, int ref_idx_curr, int ref_idx)

{

    RefPicList *refPicList = s->ref->refPicList;

    MvField *tab_mvf       = s->ref->tab_mvf;

    int ref_pic_elist      = refPicList[elist].list[TAB_MVF(x, y).ref_idx[elist]];

    int ref_pic_curr       = refPicList[ref_idx_curr].list[ref_idx];



    if (ref_pic_elist != ref_pic_curr)

        mv_scale(mv, mv, s->poc - ref_pic_elist, s->poc - ref_pic_curr);

}
