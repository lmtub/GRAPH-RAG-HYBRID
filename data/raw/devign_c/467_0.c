static void hls_prediction_unit(HEVCContext *s, int x0, int y0,

                                int nPbW, int nPbH,

                                int log2_cb_size, int partIdx)

{

#define POS(c_idx, x, y)                                                              \

    &s->frame->data[c_idx][((y) >> s->sps->vshift[c_idx]) * s->frame->linesize[c_idx] + \

                           (((x) >> s->sps->hshift[c_idx]) << s->sps->pixel_shift)]

    HEVCLocalContext *lc = &s->HEVClc;

    int merge_idx = 0;

    struct MvField current_mv = {{{ 0 }}};



    int min_pu_width = s->sps->min_pu_width;



    MvField *tab_mvf = s->ref->tab_mvf;

    RefPicList  *refPicList = s->ref->refPicList;

    HEVCFrame *ref0, *ref1;



    int tmpstride = MAX_PB_SIZE;



    uint8_t *dst0 = POS(0, x0, y0);

    uint8_t *dst1 = POS(1, x0, y0);

    uint8_t *dst2 = POS(2, x0, y0);

    int log2_min_cb_size = s->sps->log2_min_cb_size;

    int min_cb_width     = s->sps->min_cb_width;

    int x_cb             = x0 >> log2_min_cb_size;

    int y_cb             = y0 >> log2_min_cb_size;

    int x_pu, y_pu;

    int i, j;



    int skip_flag = SAMPLE_CTB(s->skip_flag, x_cb, y_cb);



    if (!skip_flag)

        lc->pu.merge_flag = ff_hevc_merge_flag_decode(s);



    if (skip_flag || lc->pu.merge_flag) {

        if (s->sh.max_num_merge_cand > 1)

            merge_idx = ff_hevc_merge_idx_decode(s);

        else

            merge_idx = 0;



        ff_hevc_luma_mv_merge_mode(s, x0, y0, nPbW, nPbH, log2_cb_size,

                                   partIdx, merge_idx, &current_mv);

    } else {

        enum InterPredIdc inter_pred_idc = PRED_L0;

        int mvp_flag;



        ff_hevc_set_neighbour_available(s, x0, y0, nPbW, nPbH);

        if (s->sh.slice_type == B_SLICE)

            inter_pred_idc = ff_hevc_inter_pred_idc_decode(s, nPbW, nPbH);



        if (inter_pred_idc != PRED_L1) {

            if (s->sh.nb_refs[L0]) {

                current_mv.ref_idx[0]= ff_hevc_ref_idx_lx_decode(s, s->sh.nb_refs[L0]);

            }

            current_mv.pred_flag[0] = 1;

            hls_mvd_coding(s, x0, y0, 0);

            mvp_flag = ff_hevc_mvp_lx_flag_decode(s);

            ff_hevc_luma_mv_mvp_mode(s, x0, y0, nPbW, nPbH, log2_cb_size,

                                     partIdx, merge_idx, &current_mv,

                                     mvp_flag, 0);

            current_mv.mv[0].x += lc->pu.mvd.x;

            current_mv.mv[0].y += lc->pu.mvd.y;

        }



        if (inter_pred_idc != PRED_L0) {

            if (s->sh.nb_refs[L1]) {

                current_mv.ref_idx[1]= ff_hevc_ref_idx_lx_decode(s, s->sh.nb_refs[L1]);

            }



            if (s->sh.mvd_l1_zero_flag == 1 && inter_pred_idc == PRED_BI) {

                AV_ZERO32(&lc->pu.mvd);

            } else {

                hls_mvd_coding(s, x0, y0, 1);

            }



            current_mv.pred_flag[1] = 1;

            mvp_flag = ff_hevc_mvp_lx_flag_decode(s);

            ff_hevc_luma_mv_mvp_mode(s, x0, y0, nPbW, nPbH, log2_cb_size,

                                     partIdx, merge_idx, &current_mv,

                                     mvp_flag, 1);

            current_mv.mv[1].x += lc->pu.mvd.x;

            current_mv.mv[1].y += lc->pu.mvd.y;

        }

    }



    x_pu = x0 >> s->sps->log2_min_pu_size;

    y_pu = y0 >> s->sps->log2_min_pu_size;



    for (j = 0; j < nPbH >> s->sps->log2_min_pu_size; j++)

        for (i = 0; i < nPbW >> s->sps->log2_min_pu_size; i++)

            tab_mvf[(y_pu + j) * min_pu_width + x_pu + i] = current_mv;



    if (current_mv.pred_flag[0]) {

        ref0 = refPicList[0].ref[current_mv.ref_idx[0]];

        if (!ref0)

            return;

        hevc_await_progress(s, ref0, &current_mv.mv[0], y0, nPbH);

    }

    if (current_mv.pred_flag[1]) {

        ref1 = refPicList[1].ref[current_mv.ref_idx[1]];

        if (!ref1)

            return;

        hevc_await_progress(s, ref1, &current_mv.mv[1], y0, nPbH);

    }



    if (current_mv.pred_flag[0] && !current_mv.pred_flag[1]) {

        DECLARE_ALIGNED(16, int16_t,  tmp[MAX_PB_SIZE * MAX_PB_SIZE]);

        DECLARE_ALIGNED(16, int16_t, tmp2[MAX_PB_SIZE * MAX_PB_SIZE]);



        luma_mc(s, tmp, tmpstride, ref0->frame,

                &current_mv.mv[0], x0, y0, nPbW, nPbH);



        if ((s->sh.slice_type == P_SLICE && s->pps->weighted_pred_flag) ||

            (s->sh.slice_type == B_SLICE && s->pps->weighted_bipred_flag)) {

            s->hevcdsp.weighted_pred(s->sh.luma_log2_weight_denom,

                                     s->sh.luma_weight_l0[current_mv.ref_idx[0]],

                                     s->sh.luma_offset_l0[current_mv.ref_idx[0]],

                                     dst0, s->frame->linesize[0], tmp,

                                     tmpstride, nPbW, nPbH);

        } else {

            s->hevcdsp.put_unweighted_pred(dst0, s->frame->linesize[0], tmp, tmpstride, nPbW, nPbH);

        }

        chroma_mc(s, tmp, tmp2, tmpstride, ref0->frame,

                  &current_mv.mv[0], x0 / 2, y0 / 2, nPbW / 2, nPbH / 2);



        if ((s->sh.slice_type == P_SLICE && s->pps->weighted_pred_flag) ||

            (s->sh.slice_type == B_SLICE && s->pps->weighted_bipred_flag)) {

            s->hevcdsp.weighted_pred(s->sh.chroma_log2_weight_denom,

                                     s->sh.chroma_weight_l0[current_mv.ref_idx[0]][0],

                                     s->sh.chroma_offset_l0[current_mv.ref_idx[0]][0],

                                     dst1, s->frame->linesize[1], tmp, tmpstride,

                                     nPbW / 2, nPbH / 2);

            s->hevcdsp.weighted_pred(s->sh.chroma_log2_weight_denom,

                                     s->sh.chroma_weight_l0[current_mv.ref_idx[0]][1],

                                     s->sh.chroma_offset_l0[current_mv.ref_idx[0]][1],

                                     dst2, s->frame->linesize[2], tmp2, tmpstride,

                                     nPbW / 2, nPbH / 2);

        } else {

            s->hevcdsp.put_unweighted_pred(dst1, s->frame->linesize[1], tmp, tmpstride, nPbW/2, nPbH/2);

            s->hevcdsp.put_unweighted_pred(dst2, s->frame->linesize[2], tmp2, tmpstride, nPbW/2, nPbH/2);

        }

    } else if (!current_mv.pred_flag[0] && current_mv.pred_flag[1]) {

        DECLARE_ALIGNED(16, int16_t, tmp [MAX_PB_SIZE * MAX_PB_SIZE]);

        DECLARE_ALIGNED(16, int16_t, tmp2[MAX_PB_SIZE * MAX_PB_SIZE]);



        if (!ref1)

            return;



        luma_mc(s, tmp, tmpstride, ref1->frame,

                &current_mv.mv[1], x0, y0, nPbW, nPbH);



        if ((s->sh.slice_type == P_SLICE && s->pps->weighted_pred_flag) ||

            (s->sh.slice_type == B_SLICE && s->pps->weighted_bipred_flag)) {

            s->hevcdsp.weighted_pred(s->sh.luma_log2_weight_denom,

                                      s->sh.luma_weight_l1[current_mv.ref_idx[1]],

                                      s->sh.luma_offset_l1[current_mv.ref_idx[1]],

                                      dst0, s->frame->linesize[0], tmp, tmpstride,

                                      nPbW, nPbH);

        } else {

            s->hevcdsp.put_unweighted_pred(dst0, s->frame->linesize[0], tmp, tmpstride, nPbW, nPbH);

        }



        chroma_mc(s, tmp, tmp2, tmpstride, ref1->frame,

                  &current_mv.mv[1], x0/2, y0/2, nPbW/2, nPbH/2);



        if ((s->sh.slice_type == P_SLICE && s->pps->weighted_pred_flag) ||

            (s->sh.slice_type == B_SLICE && s->pps->weighted_bipred_flag)) {

            s->hevcdsp.weighted_pred(s->sh.chroma_log2_weight_denom,

                                     s->sh.chroma_weight_l1[current_mv.ref_idx[1]][0],

                                     s->sh.chroma_offset_l1[current_mv.ref_idx[1]][0],

                                     dst1, s->frame->linesize[1], tmp, tmpstride, nPbW/2, nPbH/2);

            s->hevcdsp.weighted_pred(s->sh.chroma_log2_weight_denom,

                                     s->sh.chroma_weight_l1[current_mv.ref_idx[1]][1],

                                     s->sh.chroma_offset_l1[current_mv.ref_idx[1]][1],

                                     dst2, s->frame->linesize[2], tmp2, tmpstride, nPbW/2, nPbH/2);

        } else {

            s->hevcdsp.put_unweighted_pred(dst1, s->frame->linesize[1], tmp, tmpstride, nPbW/2, nPbH/2);

            s->hevcdsp.put_unweighted_pred(dst2, s->frame->linesize[2], tmp2, tmpstride, nPbW/2, nPbH/2);

        }

    } else if (current_mv.pred_flag[0] && current_mv.pred_flag[1]) {

        DECLARE_ALIGNED(16, int16_t, tmp [MAX_PB_SIZE * MAX_PB_SIZE]);

        DECLARE_ALIGNED(16, int16_t, tmp2[MAX_PB_SIZE * MAX_PB_SIZE]);

        DECLARE_ALIGNED(16, int16_t, tmp3[MAX_PB_SIZE * MAX_PB_SIZE]);

        DECLARE_ALIGNED(16, int16_t, tmp4[MAX_PB_SIZE * MAX_PB_SIZE]);

        HEVCFrame *ref0 = refPicList[0].ref[current_mv.ref_idx[0]];

        HEVCFrame *ref1 = refPicList[1].ref[current_mv.ref_idx[1]];



        if (!ref0 || !ref1)

            return;



        luma_mc(s, tmp, tmpstride, ref0->frame,

                &current_mv.mv[0], x0, y0, nPbW, nPbH);

        luma_mc(s, tmp2, tmpstride, ref1->frame,

                &current_mv.mv[1], x0, y0, nPbW, nPbH);



        if ((s->sh.slice_type == P_SLICE && s->pps->weighted_pred_flag) ||

            (s->sh.slice_type == B_SLICE && s->pps->weighted_bipred_flag)) {

            s->hevcdsp.weighted_pred_avg(s->sh.luma_log2_weight_denom,

                                         s->sh.luma_weight_l0[current_mv.ref_idx[0]],

                                         s->sh.luma_weight_l1[current_mv.ref_idx[1]],

                                         s->sh.luma_offset_l0[current_mv.ref_idx[0]],

                                         s->sh.luma_offset_l1[current_mv.ref_idx[1]],

                                         dst0, s->frame->linesize[0],

                                         tmp, tmp2, tmpstride, nPbW, nPbH);

        } else {

            s->hevcdsp.put_weighted_pred_avg(dst0, s->frame->linesize[0],

                                             tmp, tmp2, tmpstride, nPbW, nPbH);

        }



        chroma_mc(s, tmp, tmp2, tmpstride, ref0->frame,

                  &current_mv.mv[0], x0 / 2, y0 / 2, nPbW / 2, nPbH / 2);

        chroma_mc(s, tmp3, tmp4, tmpstride, ref1->frame,

                  &current_mv.mv[1], x0 / 2, y0 / 2, nPbW / 2, nPbH / 2);



        if ((s->sh.slice_type == P_SLICE && s->pps->weighted_pred_flag) ||

            (s->sh.slice_type == B_SLICE && s->pps->weighted_bipred_flag)) {

            s->hevcdsp.weighted_pred_avg(s->sh.chroma_log2_weight_denom,

                                         s->sh.chroma_weight_l0[current_mv.ref_idx[0]][0],

                                         s->sh.chroma_weight_l1[current_mv.ref_idx[1]][0],

                                         s->sh.chroma_offset_l0[current_mv.ref_idx[0]][0],

                                         s->sh.chroma_offset_l1[current_mv.ref_idx[1]][0],

                                         dst1, s->frame->linesize[1], tmp, tmp3,

                                         tmpstride, nPbW / 2, nPbH / 2);

            s->hevcdsp.weighted_pred_avg(s->sh.chroma_log2_weight_denom,

                                         s->sh.chroma_weight_l0[current_mv.ref_idx[0]][1],

                                         s->sh.chroma_weight_l1[current_mv.ref_idx[1]][1],

                                         s->sh.chroma_offset_l0[current_mv.ref_idx[0]][1],

                                         s->sh.chroma_offset_l1[current_mv.ref_idx[1]][1],

                                         dst2, s->frame->linesize[2], tmp2, tmp4,

                                         tmpstride, nPbW / 2, nPbH / 2);

        } else {

            s->hevcdsp.put_weighted_pred_avg(dst1, s->frame->linesize[1], tmp, tmp3, tmpstride, nPbW/2, nPbH/2);

            s->hevcdsp.put_weighted_pred_avg(dst2, s->frame->linesize[2], tmp2, tmp4, tmpstride, nPbW/2, nPbH/2);

        }

    }

}
