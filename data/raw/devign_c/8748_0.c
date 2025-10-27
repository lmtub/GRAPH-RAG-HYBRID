RefPicList *ff_hevc_get_ref_list(HEVCContext *s, HEVCFrame *ref, int x0, int y0)

{

    if (x0 < 0 || y0 < 0) {

        return s->ref->refPicList;

    } else {

        int x_cb         = x0 >> s->sps->log2_ctb_size;

        int y_cb         = y0 >> s->sps->log2_ctb_size;

        int pic_width_cb = (s->sps->width + (1 << s->sps->log2_ctb_size) - 1) >>

                           s->sps->log2_ctb_size;

        int ctb_addr_ts  = s->pps->ctb_addr_rs_to_ts[y_cb * pic_width_cb + x_cb];

        return (RefPicList *)ref->rpl_tab[ctb_addr_ts];

    }

}
