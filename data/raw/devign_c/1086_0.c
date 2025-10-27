static void hls_transform_unit(HEVCContext *s, int x0, int y0,

                               int xBase, int yBase, int cb_xBase, int cb_yBase,

                               int log2_cb_size, int log2_trafo_size,

                               int trafo_depth, int blk_idx)

{

    HEVCLocalContext *lc = &s->HEVClc;



    if (lc->cu.pred_mode == MODE_INTRA) {

        int trafo_size = 1 << log2_trafo_size;

        ff_hevc_set_neighbour_available(s, x0, y0, trafo_size, trafo_size);



        s->hpc.intra_pred(s, x0, y0, log2_trafo_size, 0);

        if (log2_trafo_size > 2) {

            trafo_size = trafo_size << (s->sps->hshift[1] - 1);

            ff_hevc_set_neighbour_available(s, x0, y0, trafo_size, trafo_size);

            s->hpc.intra_pred(s, x0, y0, log2_trafo_size - 1, 1);

            s->hpc.intra_pred(s, x0, y0, log2_trafo_size - 1, 2);

        } else if (blk_idx == 3) {

            trafo_size = trafo_size << s->sps->hshift[1];

            ff_hevc_set_neighbour_available(s, xBase, yBase,

                                            trafo_size, trafo_size);

            s->hpc.intra_pred(s, xBase, yBase, log2_trafo_size, 1);

            s->hpc.intra_pred(s, xBase, yBase, log2_trafo_size, 2);

        }

    }



    if (lc->tt.cbf_luma ||

        SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth], x0, y0) ||

        SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth], x0, y0)) {

        int scan_idx   = SCAN_DIAG;

        int scan_idx_c = SCAN_DIAG;



        if (s->pps->cu_qp_delta_enabled_flag && !lc->tu.is_cu_qp_delta_coded) {

            lc->tu.cu_qp_delta = ff_hevc_cu_qp_delta_abs(s);

            if (lc->tu.cu_qp_delta != 0)

                if (ff_hevc_cu_qp_delta_sign_flag(s) == 1)

                    lc->tu.cu_qp_delta = -lc->tu.cu_qp_delta;

            lc->tu.is_cu_qp_delta_coded = 1;

            ff_hevc_set_qPy(s, x0, y0, cb_xBase, cb_yBase, log2_cb_size);

        }



        if (lc->cu.pred_mode == MODE_INTRA && log2_trafo_size < 4) {

            if (lc->tu.cur_intra_pred_mode >= 6 &&

                lc->tu.cur_intra_pred_mode <= 14) {

                scan_idx = SCAN_VERT;

            } else if (lc->tu.cur_intra_pred_mode >= 22 &&

                       lc->tu.cur_intra_pred_mode <= 30) {

                scan_idx = SCAN_HORIZ;

            }



            if (lc->pu.intra_pred_mode_c >=  6 &&

                lc->pu.intra_pred_mode_c <= 14) {

                scan_idx_c = SCAN_VERT;

            } else if (lc->pu.intra_pred_mode_c >= 22 &&

                       lc->pu.intra_pred_mode_c <= 30) {

                scan_idx_c = SCAN_HORIZ;

            }

        }



        if (lc->tt.cbf_luma)

            hls_residual_coding(s, x0, y0, log2_trafo_size, scan_idx, 0);

        if (log2_trafo_size > 2) {

            if (SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth], x0, y0))

                hls_residual_coding(s, x0, y0, log2_trafo_size - 1, scan_idx_c, 1);

            if (SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth], x0, y0))

                hls_residual_coding(s, x0, y0, log2_trafo_size - 1, scan_idx_c, 2);

        } else if (blk_idx == 3) {

            if (SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth], xBase, yBase))

                hls_residual_coding(s, xBase, yBase, log2_trafo_size, scan_idx_c, 1);

            if (SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth], xBase, yBase))

                hls_residual_coding(s, xBase, yBase, log2_trafo_size, scan_idx_c, 2);

        }

    }

}
