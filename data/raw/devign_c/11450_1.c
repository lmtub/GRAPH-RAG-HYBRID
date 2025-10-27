static int hls_decode_entry(AVCodecContext *avctxt, void *isFilterThread)

{

    HEVCContext *s  = avctxt->priv_data;

    int ctb_size    = 1 << s->ps.sps->log2_ctb_size;

    int more_data   = 1;

    int x_ctb       = 0;

    int y_ctb       = 0;

    int ctb_addr_ts = s->ps.pps->ctb_addr_rs_to_ts[s->sh.slice_ctb_addr_rs];



    if (!ctb_addr_ts && s->sh.dependent_slice_segment_flag) {

        av_log(s->avctx, AV_LOG_ERROR, "Impossible initial tile.\n");

        return AVERROR_INVALIDDATA;

    }



    if (s->sh.dependent_slice_segment_flag) {

        int prev_rs = s->ps.pps->ctb_addr_ts_to_rs[ctb_addr_ts - 1];

        if (s->tab_slice_address[prev_rs] != s->sh.slice_addr) {

            av_log(s->avctx, AV_LOG_ERROR, "Previous slice segment missing\n");

            return AVERROR_INVALIDDATA;

        }

    }



    while (more_data && ctb_addr_ts < s->ps.sps->ctb_size) {

        int ctb_addr_rs = s->ps.pps->ctb_addr_ts_to_rs[ctb_addr_ts];



        x_ctb = (ctb_addr_rs % ((s->ps.sps->width + ctb_size - 1) >> s->ps.sps->log2_ctb_size)) << s->ps.sps->log2_ctb_size;

        y_ctb = (ctb_addr_rs / ((s->ps.sps->width + ctb_size - 1) >> s->ps.sps->log2_ctb_size)) << s->ps.sps->log2_ctb_size;

        hls_decode_neighbour(s, x_ctb, y_ctb, ctb_addr_ts);



        ff_hevc_cabac_init(s, ctb_addr_ts);



        hls_sao_param(s, x_ctb >> s->ps.sps->log2_ctb_size, y_ctb >> s->ps.sps->log2_ctb_size);



        s->deblock[ctb_addr_rs].beta_offset = s->sh.beta_offset;

        s->deblock[ctb_addr_rs].tc_offset   = s->sh.tc_offset;

        s->filter_slice_edges[ctb_addr_rs]  = s->sh.slice_loop_filter_across_slices_enabled_flag;



        more_data = hls_coding_quadtree(s, x_ctb, y_ctb, s->ps.sps->log2_ctb_size, 0);

        if (more_data < 0) {

            s->tab_slice_address[ctb_addr_rs] = -1;

            return more_data;

        }





        ctb_addr_ts++;

        ff_hevc_save_states(s, ctb_addr_ts);

        ff_hevc_hls_filters(s, x_ctb, y_ctb, ctb_size);

    }



    if (x_ctb + ctb_size >= s->ps.sps->width &&

        y_ctb + ctb_size >= s->ps.sps->height)

        ff_hevc_hls_filter(s, x_ctb, y_ctb, ctb_size);



    return ctb_addr_ts;

}
