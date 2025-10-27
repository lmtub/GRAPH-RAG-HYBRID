void ff_hevc_hls_residual_coding(HEVCContext *s, int x0, int y0,

                                int log2_trafo_size, enum ScanType scan_idx,

                                int c_idx)

{

#define GET_COORD(offset, n)                                    \

    do {                                                        \

        x_c = (x_cg << 2) + scan_x_off[n];                      \

        y_c = (y_cg << 2) + scan_y_off[n];                      \

    } while (0)

    HEVCLocalContext *lc = s->HEVClc;

    int transform_skip_flag = 0;



    int last_significant_coeff_x, last_significant_coeff_y;

    int last_scan_pos;

    int n_end;

    int num_coeff = 0;

    int greater1_ctx = 1;



    int num_last_subset;

    int x_cg_last_sig, y_cg_last_sig;



    const uint8_t *scan_x_cg, *scan_y_cg, *scan_x_off, *scan_y_off;



    ptrdiff_t stride = s->frame->linesize[c_idx];

    int hshift = s->sps->hshift[c_idx];

    int vshift = s->sps->vshift[c_idx];

    uint8_t *dst = &s->frame->data[c_idx][(y0 >> vshift) * stride +

                                          ((x0 >> hshift) << s->sps->pixel_shift)];

    int16_t *coeffs = lc->tu.coeffs[c_idx > 0];

    uint8_t significant_coeff_group_flag[8][8] = {{0}};

    int explicit_rdpcm_flag = 0;

    int explicit_rdpcm_dir_flag;



    int trafo_size = 1 << log2_trafo_size;

    int i;

    int qp,shift,add,scale,scale_m;

    const uint8_t level_scale[] = { 40, 45, 51, 57, 64, 72 };

    const uint8_t *scale_matrix = NULL;

    uint8_t dc_scale;

    int pred_mode_intra = (c_idx == 0) ? lc->tu.intra_pred_mode :

                                         lc->tu.intra_pred_mode_c;



    memset(coeffs, 0, trafo_size * trafo_size * sizeof(int16_t));



    // Derive QP for dequant

    if (!lc->cu.cu_transquant_bypass_flag) {

        static const int qp_c[] = { 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37 };

        static const uint8_t rem6[51 + 4 * 6 + 1] = {

            0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2,

            3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,

            0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,

            4, 5, 0, 1, 2, 3, 4, 5, 0, 1

        };



        static const uint8_t div6[51 + 4 * 6 + 1] = {

            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3,  3,  3,

            3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,  6,  6,

            7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10,

            10, 10, 11, 11, 11, 11, 11, 11, 12, 12

        };

        int qp_y = lc->qp_y;



        if (s->pps->transform_skip_enabled_flag &&

            log2_trafo_size <= s->pps->log2_max_transform_skip_block_size) {

            transform_skip_flag = ff_hevc_transform_skip_flag_decode(s, c_idx);

        }



        if (c_idx == 0) {

            qp = qp_y + s->sps->qp_bd_offset;

        } else {

            int qp_i, offset;



            if (c_idx == 1)

                offset = s->pps->cb_qp_offset + s->sh.slice_cb_qp_offset +

                         lc->tu.cu_qp_offset_cb;

            else

                offset = s->pps->cr_qp_offset + s->sh.slice_cr_qp_offset +

                         lc->tu.cu_qp_offset_cr;



            qp_i = av_clip(qp_y + offset, - s->sps->qp_bd_offset, 57);

            if (s->sps->chroma_format_idc == 1) {

                if (qp_i < 30)

                    qp = qp_i;

                else if (qp_i > 43)

                    qp = qp_i - 6;

                else

                    qp = qp_c[qp_i - 30];

            } else {

                if (qp_i > 51)

                    qp = 51;

                else

                    qp = qp_i;

            }



            qp += s->sps->qp_bd_offset;

        }



        shift    = s->sps->bit_depth + log2_trafo_size - 5;

        add      = 1 << (shift-1);

        scale    = level_scale[rem6[qp]] << (div6[qp]);

        scale_m  = 16; // default when no custom scaling lists.

        dc_scale = 16;



        if (s->sps->scaling_list_enable_flag && !(transform_skip_flag && log2_trafo_size > 2)) {

            const ScalingList *sl = s->pps->scaling_list_data_present_flag ?

            &s->pps->scaling_list : &s->sps->scaling_list;

            int matrix_id = lc->cu.pred_mode != MODE_INTRA;



            matrix_id = 3 * matrix_id + c_idx;



            scale_matrix = sl->sl[log2_trafo_size - 2][matrix_id];

            if (log2_trafo_size >= 4)

                dc_scale = sl->sl_dc[log2_trafo_size - 4][matrix_id];

        }

    } else {

        shift        = 0;

        add          = 0;

        scale        = 0;

        dc_scale     = 0;

    }



    if (lc->cu.pred_mode == MODE_INTER && s->sps->explicit_rdpcm_enabled_flag &&

        (transform_skip_flag || lc->cu.cu_transquant_bypass_flag)) {

        explicit_rdpcm_flag = explicit_rdpcm_flag_decode(s, c_idx);

        if (explicit_rdpcm_flag) {

            explicit_rdpcm_dir_flag = explicit_rdpcm_dir_flag_decode(s, c_idx);

        }

    }



    last_significant_coeff_xy_prefix_decode(s, c_idx, log2_trafo_size,

                                           &last_significant_coeff_x, &last_significant_coeff_y);



    if (last_significant_coeff_x > 3) {

        int suffix = last_significant_coeff_suffix_decode(s, last_significant_coeff_x);

        last_significant_coeff_x = (1 << ((last_significant_coeff_x >> 1) - 1)) *

        (2 + (last_significant_coeff_x & 1)) +

        suffix;

    }



    if (last_significant_coeff_y > 3) {

        int suffix = last_significant_coeff_suffix_decode(s, last_significant_coeff_y);

        last_significant_coeff_y = (1 << ((last_significant_coeff_y >> 1) - 1)) *

        (2 + (last_significant_coeff_y & 1)) +

        suffix;

    }



    if (scan_idx == SCAN_VERT)

        FFSWAP(int, last_significant_coeff_x, last_significant_coeff_y);



    x_cg_last_sig = last_significant_coeff_x >> 2;

    y_cg_last_sig = last_significant_coeff_y >> 2;



    switch (scan_idx) {

    case SCAN_DIAG: {

        int last_x_c = last_significant_coeff_x & 3;

        int last_y_c = last_significant_coeff_y & 3;



        scan_x_off = ff_hevc_diag_scan4x4_x;

        scan_y_off = ff_hevc_diag_scan4x4_y;

        num_coeff = diag_scan4x4_inv[last_y_c][last_x_c];

        if (trafo_size == 4) {

            scan_x_cg = scan_1x1;

            scan_y_cg = scan_1x1;

        } else if (trafo_size == 8) {

            num_coeff += diag_scan2x2_inv[y_cg_last_sig][x_cg_last_sig] << 4;

            scan_x_cg = diag_scan2x2_x;

            scan_y_cg = diag_scan2x2_y;

        } else if (trafo_size == 16) {

            num_coeff += diag_scan4x4_inv[y_cg_last_sig][x_cg_last_sig] << 4;

            scan_x_cg = ff_hevc_diag_scan4x4_x;

            scan_y_cg = ff_hevc_diag_scan4x4_y;

        } else { // trafo_size == 32

            num_coeff += diag_scan8x8_inv[y_cg_last_sig][x_cg_last_sig] << 4;

            scan_x_cg = ff_hevc_diag_scan8x8_x;

            scan_y_cg = ff_hevc_diag_scan8x8_y;

        }

        break;

    }

    case SCAN_HORIZ:

        scan_x_cg = horiz_scan2x2_x;

        scan_y_cg = horiz_scan2x2_y;

        scan_x_off = horiz_scan4x4_x;

        scan_y_off = horiz_scan4x4_y;

        num_coeff = horiz_scan8x8_inv[last_significant_coeff_y][last_significant_coeff_x];

        break;

    default: //SCAN_VERT

        scan_x_cg = horiz_scan2x2_y;

        scan_y_cg = horiz_scan2x2_x;

        scan_x_off = horiz_scan4x4_y;

        scan_y_off = horiz_scan4x4_x;

        num_coeff = horiz_scan8x8_inv[last_significant_coeff_x][last_significant_coeff_y];

        break;

    }

    num_coeff++;

    num_last_subset = (num_coeff - 1) >> 4;



    for (i = num_last_subset; i >= 0; i--) {

        int n, m;

        int x_cg, y_cg, x_c, y_c, pos;

        int implicit_non_zero_coeff = 0;

        int64_t trans_coeff_level;

        int prev_sig = 0;

        int offset = i << 4;

        int rice_init = 0;



        uint8_t significant_coeff_flag_idx[16];

        uint8_t nb_significant_coeff_flag = 0;



        x_cg = scan_x_cg[i];

        y_cg = scan_y_cg[i];



        if ((i < num_last_subset) && (i > 0)) {

            int ctx_cg = 0;

            if (x_cg < (1 << (log2_trafo_size - 2)) - 1)

                ctx_cg += significant_coeff_group_flag[x_cg + 1][y_cg];

            if (y_cg < (1 << (log2_trafo_size - 2)) - 1)

                ctx_cg += significant_coeff_group_flag[x_cg][y_cg + 1];



            significant_coeff_group_flag[x_cg][y_cg] =

                significant_coeff_group_flag_decode(s, c_idx, ctx_cg);

            implicit_non_zero_coeff = 1;

        } else {

            significant_coeff_group_flag[x_cg][y_cg] =

            ((x_cg == x_cg_last_sig && y_cg == y_cg_last_sig) ||

             (x_cg == 0 && y_cg == 0));

        }



        last_scan_pos = num_coeff - offset - 1;



        if (i == num_last_subset) {

            n_end = last_scan_pos - 1;

            significant_coeff_flag_idx[0] = last_scan_pos;

            nb_significant_coeff_flag = 1;

        } else {

            n_end = 15;

        }



        if (x_cg < ((1 << log2_trafo_size) - 1) >> 2)

            prev_sig = !!significant_coeff_group_flag[x_cg + 1][y_cg];

        if (y_cg < ((1 << log2_trafo_size) - 1) >> 2)

            prev_sig += (!!significant_coeff_group_flag[x_cg][y_cg + 1] << 1);



        if (significant_coeff_group_flag[x_cg][y_cg] && n_end >= 0) {

            static const uint8_t ctx_idx_map[] = {

                0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8, // log2_trafo_size == 2

                1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, // prev_sig == 0

                2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, // prev_sig == 1

                2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, // prev_sig == 2

                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2  // default

            };

            const uint8_t *ctx_idx_map_p;

            int scf_offset = 0;

            if (s->sps->transform_skip_context_enabled_flag &&

                (transform_skip_flag || lc->cu.cu_transquant_bypass_flag)) {

                ctx_idx_map_p = (uint8_t*) &ctx_idx_map[4 * 16];

                if (c_idx == 0) {

                    scf_offset = 40;

                } else {

                    scf_offset = 14 + 27;

                }

            } else {

                if (c_idx != 0)

                    scf_offset = 27;

                if (log2_trafo_size == 2) {

                    ctx_idx_map_p = (uint8_t*) &ctx_idx_map[0];

                } else {

                    ctx_idx_map_p = (uint8_t*) &ctx_idx_map[(prev_sig + 1) << 4];

                    if (c_idx == 0) {

                        if ((x_cg > 0 || y_cg > 0))

                            scf_offset += 3;

                        if (log2_trafo_size == 3) {

                            scf_offset += (scan_idx == SCAN_DIAG) ? 9 : 15;

                        } else {

                            scf_offset += 21;

                        }

                    } else {

                        if (log2_trafo_size == 3)

                            scf_offset += 9;

                        else

                            scf_offset += 12;

                    }

                }

            }

            for (n = n_end; n > 0; n--) {

                x_c = scan_x_off[n];

                y_c = scan_y_off[n];

                if (significant_coeff_flag_decode(s, x_c, y_c, scf_offset, ctx_idx_map_p)) {

                    significant_coeff_flag_idx[nb_significant_coeff_flag] = n;

                    nb_significant_coeff_flag++;

                    implicit_non_zero_coeff = 0;

                }

            }

            if (implicit_non_zero_coeff == 0) {

                if (s->sps->transform_skip_context_enabled_flag &&

                    (transform_skip_flag || lc->cu.cu_transquant_bypass_flag)) {

                    if (c_idx == 0) {

                        scf_offset = 42;

                    } else {

                        scf_offset = 16 + 27;

                    }

                } else {

                    if (i == 0) {

                        if (c_idx == 0)

                            scf_offset = 0;

                        else

                            scf_offset = 27;

                    } else {

                        scf_offset = 2 + scf_offset;

                    }

                }

                if (significant_coeff_flag_decode_0(s, c_idx, scf_offset) == 1) {

                    significant_coeff_flag_idx[nb_significant_coeff_flag] = 0;

                    nb_significant_coeff_flag++;

                }

            } else {

                significant_coeff_flag_idx[nb_significant_coeff_flag] = 0;

                nb_significant_coeff_flag++;

            }

        }



        n_end = nb_significant_coeff_flag;





        if (n_end) {

            int first_nz_pos_in_cg;

            int last_nz_pos_in_cg;

            int c_rice_param = 0;

            int first_greater1_coeff_idx = -1;

            uint8_t coeff_abs_level_greater1_flag[8];

            uint16_t coeff_sign_flag;

            int sum_abs = 0;

            int sign_hidden;

            int sb_type;





            // initialize first elem of coeff_bas_level_greater1_flag

            int ctx_set = (i > 0 && c_idx == 0) ? 2 : 0;



            if (s->sps->persistent_rice_adaptation_enabled_flag) {

                if (!transform_skip_flag && !lc->cu.cu_transquant_bypass_flag)

                    sb_type = 2 * (c_idx == 0 ? 1 : 0);

                else

                    sb_type = 2 * (c_idx == 0 ? 1 : 0) + 1;

                c_rice_param = lc->stat_coeff[sb_type] / 4;

            }



            if (!(i == num_last_subset) && greater1_ctx == 0)

                ctx_set++;

            greater1_ctx = 1;

            last_nz_pos_in_cg = significant_coeff_flag_idx[0];



            for (m = 0; m < (n_end > 8 ? 8 : n_end); m++) {

                int inc = (ctx_set << 2) + greater1_ctx;

                coeff_abs_level_greater1_flag[m] =

                    coeff_abs_level_greater1_flag_decode(s, c_idx, inc);

                if (coeff_abs_level_greater1_flag[m]) {

                    greater1_ctx = 0;

                    if (first_greater1_coeff_idx == -1)

                        first_greater1_coeff_idx = m;

                } else if (greater1_ctx > 0 && greater1_ctx < 3) {

                    greater1_ctx++;

                }

            }

            first_nz_pos_in_cg = significant_coeff_flag_idx[n_end - 1];



            if (lc->cu.cu_transquant_bypass_flag ||

                (lc->cu.pred_mode ==  MODE_INTRA  &&

                 s->sps->implicit_rdpcm_enabled_flag  &&  transform_skip_flag  &&

                 (pred_mode_intra == 10 || pred_mode_intra  ==  26 )) ||

                 explicit_rdpcm_flag)

                sign_hidden = 0;

            else

                sign_hidden = (last_nz_pos_in_cg - first_nz_pos_in_cg >= 4);



            if (first_greater1_coeff_idx != -1) {

                coeff_abs_level_greater1_flag[first_greater1_coeff_idx] += coeff_abs_level_greater2_flag_decode(s, c_idx, ctx_set);

            }

            if (!s->pps->sign_data_hiding_flag || !sign_hidden ) {

                coeff_sign_flag = coeff_sign_flag_decode(s, nb_significant_coeff_flag) << (16 - nb_significant_coeff_flag);

            } else {

                coeff_sign_flag = coeff_sign_flag_decode(s, nb_significant_coeff_flag - 1) << (16 - (nb_significant_coeff_flag - 1));

            }



            for (m = 0; m < n_end; m++) {

                n = significant_coeff_flag_idx[m];

                GET_COORD(offset, n);

                if (m < 8) {

                    trans_coeff_level = 1 + coeff_abs_level_greater1_flag[m];

                    if (trans_coeff_level == ((m == first_greater1_coeff_idx) ? 3 : 2)) {

                        int last_coeff_abs_level_remaining = coeff_abs_level_remaining_decode(s, c_rice_param);



                        trans_coeff_level += last_coeff_abs_level_remaining;

                        if (trans_coeff_level > (3 << c_rice_param))

                            c_rice_param = s->sps->persistent_rice_adaptation_enabled_flag ? c_rice_param + 1 : FFMIN(c_rice_param + 1, 4);

                        if (s->sps->persistent_rice_adaptation_enabled_flag && !rice_init) {

                            int c_rice_p_init = lc->stat_coeff[sb_type] / 4;

                            if (last_coeff_abs_level_remaining >= (3 << c_rice_p_init))

                                lc->stat_coeff[sb_type]++;

                            else if (2 * last_coeff_abs_level_remaining < (1 << c_rice_p_init))

                                if (lc->stat_coeff[sb_type] > 0)

                                    lc->stat_coeff[sb_type]--;

                            rice_init = 1;

                        }

                    }

                } else {

                    int last_coeff_abs_level_remaining = coeff_abs_level_remaining_decode(s, c_rice_param);



                    trans_coeff_level = 1 + last_coeff_abs_level_remaining;

                    if (trans_coeff_level > (3 << c_rice_param))

                        c_rice_param = s->sps->persistent_rice_adaptation_enabled_flag ? c_rice_param + 1 : FFMIN(c_rice_param + 1, 4);

                    if (s->sps->persistent_rice_adaptation_enabled_flag && !rice_init) {

                        int c_rice_p_init = lc->stat_coeff[sb_type] / 4;

                        if (last_coeff_abs_level_remaining >= (3 << c_rice_p_init))

                            lc->stat_coeff[sb_type]++;

                        else if (2 * last_coeff_abs_level_remaining < (1 << c_rice_p_init))

                            if (lc->stat_coeff[sb_type] > 0)

                                lc->stat_coeff[sb_type]--;

                        rice_init = 1;

                    }

                }

                if (s->pps->sign_data_hiding_flag && sign_hidden) {

                    sum_abs += trans_coeff_level;

                    if (n == first_nz_pos_in_cg && (sum_abs&1))

                        trans_coeff_level = -trans_coeff_level;

                }

                if (coeff_sign_flag >> 15)

                    trans_coeff_level = -trans_coeff_level;

                coeff_sign_flag <<= 1;

                if(!lc->cu.cu_transquant_bypass_flag) {

                    if (s->sps->scaling_list_enable_flag && !(transform_skip_flag && log2_trafo_size > 2)) {

                        if(y_c || x_c || log2_trafo_size < 4) {

                            switch(log2_trafo_size) {

                                case 3: pos = (y_c << 3) + x_c; break;

                                case 4: pos = ((y_c >> 1) << 3) + (x_c >> 1); break;

                                case 5: pos = ((y_c >> 2) << 3) + (x_c >> 2); break;

                                default: pos = (y_c << 2) + x_c; break;

                            }

                            scale_m = scale_matrix[pos];

                        } else {

                            scale_m = dc_scale;

                        }

                    }

                    trans_coeff_level = (trans_coeff_level * (int64_t)scale * (int64_t)scale_m + add) >> shift;

                    if(trans_coeff_level < 0) {

                        if((~trans_coeff_level) & 0xFffffffffff8000)

                            trans_coeff_level = -32768;

                    } else {

                        if(trans_coeff_level & 0xffffffffffff8000)

                            trans_coeff_level = 32767;

                    }

                }

                coeffs[y_c * trafo_size + x_c] = trans_coeff_level;

            }

        }

    }



    if (lc->cu.cu_transquant_bypass_flag) {

        if (explicit_rdpcm_flag || (s->sps->implicit_rdpcm_enabled_flag &&

                                    (pred_mode_intra == 10 || pred_mode_intra == 26))) {

            int mode = s->sps->implicit_rdpcm_enabled_flag ? (pred_mode_intra == 26) : explicit_rdpcm_dir_flag;



            s->hevcdsp.transform_rdpcm(coeffs, log2_trafo_size, mode);

        }

    } else {

        if (transform_skip_flag) {

            int rot = s->sps->transform_skip_rotation_enabled_flag &&

                      log2_trafo_size == 2 &&

                      lc->cu.pred_mode == MODE_INTRA;

            if (rot) {

                for (i = 0; i < 8; i++)

                    FFSWAP(int16_t, coeffs[i], coeffs[16 - i - 1]);

            }



            s->hevcdsp.transform_skip(coeffs, log2_trafo_size);



            if (explicit_rdpcm_flag || (s->sps->implicit_rdpcm_enabled_flag &&

                                        lc->cu.pred_mode == MODE_INTRA &&

                                        (pred_mode_intra == 10 || pred_mode_intra == 26))) {

                int mode = explicit_rdpcm_flag ? explicit_rdpcm_dir_flag : (pred_mode_intra == 26);



                s->hevcdsp.transform_rdpcm(coeffs, log2_trafo_size, mode);

            }

        } else if (lc->cu.pred_mode == MODE_INTRA && c_idx == 0 && log2_trafo_size == 2) {

            s->hevcdsp.idct_4x4_luma(coeffs);

        } else {

            int max_xy = FFMAX(last_significant_coeff_x, last_significant_coeff_y);

            if (max_xy == 0)

                s->hevcdsp.idct_dc[log2_trafo_size-2](coeffs);

            else {

                int col_limit = last_significant_coeff_x + last_significant_coeff_y + 4;

                if (max_xy < 4)

                    col_limit = FFMIN(4, col_limit);

                else if (max_xy < 8)

                    col_limit = FFMIN(8, col_limit);

                else if (max_xy < 12)

                    col_limit = FFMIN(24, col_limit);

                s->hevcdsp.idct[log2_trafo_size-2](coeffs, col_limit);

            }

        }

    }

    if (lc->tu.cross_pf) {

        int16_t *coeffs_y = lc->tu.coeffs[0];



        for (i = 0; i < (trafo_size * trafo_size); i++) {

            coeffs[i] = coeffs[i] + ((lc->tu.res_scale_val * coeffs_y[i]) >> 3);

        }

    }

    s->hevcdsp.transform_add[log2_trafo_size-2](dst, coeffs, stride);

}
