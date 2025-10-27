static void decode_mode(AVCodecContext *ctx)

{

    static const uint8_t left_ctx[N_BS_SIZES] = {

        0x0, 0x8, 0x0, 0x8, 0xc, 0x8, 0xc, 0xe, 0xc, 0xe, 0xf, 0xe, 0xf

    };

    static const uint8_t above_ctx[N_BS_SIZES] = {

        0x0, 0x0, 0x8, 0x8, 0x8, 0xc, 0xc, 0xc, 0xe, 0xe, 0xe, 0xf, 0xf

    };

    static const uint8_t max_tx_for_bl_bp[N_BS_SIZES] = {

        TX_32X32, TX_32X32, TX_32X32, TX_32X32, TX_16X16, TX_16X16,

        TX_16X16, TX_8X8, TX_8X8, TX_8X8, TX_4X4, TX_4X4, TX_4X4

    };

    VP9Context *s = ctx->priv_data;

    VP9Block *b = s->b;

    int row = s->row, col = s->col, row7 = s->row7;

    enum TxfmMode max_tx = max_tx_for_bl_bp[b->bs];

    int w4 = FFMIN(s->cols - col, bwh_tab[1][b->bs][0]);

    int h4 = FFMIN(s->rows - row, bwh_tab[1][b->bs][1]), y;

    int have_a = row > 0, have_l = col > s->tiling.tile_col_start;



    if (!s->segmentation.enabled) {

        b->seg_id = 0;

    } else if (s->keyframe || s->intraonly) {

        b->seg_id = s->segmentation.update_map ?

            vp8_rac_get_tree(&s->c, vp9_segmentation_tree, s->prob.seg) : 0;

    } else if (!s->segmentation.update_map ||

               (s->segmentation.temporal &&

                vp56_rac_get_prob_branchy(&s->c,

                    s->prob.segpred[s->above_segpred_ctx[col] +

                                    s->left_segpred_ctx[row7]]))) {

        int pred = 8, x;

        uint8_t *refsegmap = s->frames[LAST_FRAME].segmentation_map;



        if (!s->last_uses_2pass)

            ff_thread_await_progress(&s->frames[LAST_FRAME].tf, row >> 3, 0);

        for (y = 0; y < h4; y++)

            for (x = 0; x < w4; x++)

                pred = FFMIN(pred, refsegmap[(y + row) * 8 * s->sb_cols + x + col]);

        av_assert1(pred < 8);

        b->seg_id = pred;



        memset(&s->above_segpred_ctx[col], 1, w4);

        memset(&s->left_segpred_ctx[row7], 1, h4);

    } else {

        b->seg_id = vp8_rac_get_tree(&s->c, vp9_segmentation_tree,

                                     s->prob.seg);



        memset(&s->above_segpred_ctx[col], 0, w4);

        memset(&s->left_segpred_ctx[row7], 0, h4);

    }

    if ((s->segmentation.enabled && s->segmentation.update_map) || s->keyframe) {

        uint8_t *segmap = s->frames[CUR_FRAME].segmentation_map;



        for (y = 0; y < h4; y++)

            memset(&segmap[(y + row) * 8 * s->sb_cols + col], b->seg_id, w4);

    }



    b->skip = s->segmentation.enabled &&

        s->segmentation.feat[b->seg_id].skip_enabled;

    if (!b->skip) {

        int c = s->left_skip_ctx[row7] + s->above_skip_ctx[col];

        b->skip = vp56_rac_get_prob(&s->c, s->prob.p.skip[c]);

        s->counts.skip[c][b->skip]++;

    }



    if (s->keyframe || s->intraonly) {

        b->intra = 1;

    } else if (s->segmentation.feat[b->seg_id].ref_enabled) {

        b->intra = !s->segmentation.feat[b->seg_id].ref_val;

    } else {

        int c, bit;



        if (have_a && have_l) {

            c = s->above_intra_ctx[col] + s->left_intra_ctx[row7];

            c += (c == 2);

        } else {

            c = have_a ? 2 * s->above_intra_ctx[col] :

                have_l ? 2 * s->left_intra_ctx[row7] : 0;

        }

        bit = vp56_rac_get_prob(&s->c, s->prob.p.intra[c]);

        s->counts.intra[c][bit]++;

        b->intra = !bit;

    }



    if ((b->intra || !b->skip) && s->txfmmode == TX_SWITCHABLE) {

        int c;

        if (have_a) {

            if (have_l) {

                c = (s->above_skip_ctx[col] ? max_tx :

                     s->above_txfm_ctx[col]) +

                    (s->left_skip_ctx[row7] ? max_tx :

                     s->left_txfm_ctx[row7]) > max_tx;

            } else {

                c = s->above_skip_ctx[col] ? 1 :

                    (s->above_txfm_ctx[col] * 2 > max_tx);

            }

        } else if (have_l) {

            c = s->left_skip_ctx[row7] ? 1 :

                (s->left_txfm_ctx[row7] * 2 > max_tx);

        } else {

            c = 1;

        }

        switch (max_tx) {

        case TX_32X32:

            b->tx = vp56_rac_get_prob(&s->c, s->prob.p.tx32p[c][0]);

            if (b->tx) {

                b->tx += vp56_rac_get_prob(&s->c, s->prob.p.tx32p[c][1]);

                if (b->tx == 2)

                    b->tx += vp56_rac_get_prob(&s->c, s->prob.p.tx32p[c][2]);

            }

            s->counts.tx32p[c][b->tx]++;

            break;

        case TX_16X16:

            b->tx = vp56_rac_get_prob(&s->c, s->prob.p.tx16p[c][0]);

            if (b->tx)

                b->tx += vp56_rac_get_prob(&s->c, s->prob.p.tx16p[c][1]);

            s->counts.tx16p[c][b->tx]++;

            break;

        case TX_8X8:

            b->tx = vp56_rac_get_prob(&s->c, s->prob.p.tx8p[c]);

            s->counts.tx8p[c][b->tx]++;

            break;

        case TX_4X4:

            b->tx = TX_4X4;

            break;

        }

    } else {

        b->tx = FFMIN(max_tx, s->txfmmode);

    }



    if (s->keyframe || s->intraonly) {

        uint8_t *a = &s->above_mode_ctx[col * 2];

        uint8_t *l = &s->left_mode_ctx[(row7) << 1];



        b->comp = 0;

        if (b->bs > BS_8x8) {

            // FIXME the memory storage intermediates here aren't really

            // necessary, they're just there to make the code slightly

            // simpler for now

            b->mode[0] = a[0] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                    vp9_default_kf_ymode_probs[a[0]][l[0]]);

            if (b->bs != BS_8x4) {

                b->mode[1] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                 vp9_default_kf_ymode_probs[a[1]][b->mode[0]]);

                l[0] = a[1] = b->mode[1];

            } else {

                l[0] = a[1] = b->mode[1] = b->mode[0];

            }

            if (b->bs != BS_4x8) {

                b->mode[2] = a[0] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                        vp9_default_kf_ymode_probs[a[0]][l[1]]);

                if (b->bs != BS_8x4) {

                    b->mode[3] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                  vp9_default_kf_ymode_probs[a[1]][b->mode[2]]);

                    l[1] = a[1] = b->mode[3];

                } else {

                    l[1] = a[1] = b->mode[3] = b->mode[2];

                }

            } else {

                b->mode[2] = b->mode[0];

                l[1] = a[1] = b->mode[3] = b->mode[1];

            }

        } else {

            b->mode[0] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                          vp9_default_kf_ymode_probs[*a][*l]);

            b->mode[3] = b->mode[2] = b->mode[1] = b->mode[0];

            // FIXME this can probably be optimized

            memset(a, b->mode[0], bwh_tab[0][b->bs][0]);

            memset(l, b->mode[0], bwh_tab[0][b->bs][1]);

        }

        b->uvmode = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                     vp9_default_kf_uvmode_probs[b->mode[3]]);

    } else if (b->intra) {

        b->comp = 0;

        if (b->bs > BS_8x8) {

            b->mode[0] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                          s->prob.p.y_mode[0]);

            s->counts.y_mode[0][b->mode[0]]++;

            if (b->bs != BS_8x4) {

                b->mode[1] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                              s->prob.p.y_mode[0]);

                s->counts.y_mode[0][b->mode[1]]++;

            } else {

                b->mode[1] = b->mode[0];

            }

            if (b->bs != BS_4x8) {

                b->mode[2] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                              s->prob.p.y_mode[0]);

                s->counts.y_mode[0][b->mode[2]]++;

                if (b->bs != BS_8x4) {

                    b->mode[3] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                                  s->prob.p.y_mode[0]);

                    s->counts.y_mode[0][b->mode[3]]++;

                } else {

                    b->mode[3] = b->mode[2];

                }

            } else {

                b->mode[2] = b->mode[0];

                b->mode[3] = b->mode[1];

            }

        } else {

            static const uint8_t size_group[10] = {

                3, 3, 3, 3, 2, 2, 2, 1, 1, 1

            };

            int sz = size_group[b->bs];



            b->mode[0] = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                          s->prob.p.y_mode[sz]);

            b->mode[1] = b->mode[2] = b->mode[3] = b->mode[0];

            s->counts.y_mode[sz][b->mode[3]]++;

        }

        b->uvmode = vp8_rac_get_tree(&s->c, vp9_intramode_tree,

                                     s->prob.p.uv_mode[b->mode[3]]);

        s->counts.uv_mode[b->mode[3]][b->uvmode]++;

    } else {

        static const uint8_t inter_mode_ctx_lut[14][14] = {

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5 },

            { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 1, 3 },

            { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 1, 3 },

            { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 0, 3 },

            { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 4 },

        };



        if (s->segmentation.feat[b->seg_id].ref_enabled) {

            av_assert2(s->segmentation.feat[b->seg_id].ref_val != 0);

            b->comp = 0;

            b->ref[0] = s->segmentation.feat[b->seg_id].ref_val - 1;

        } else {

            // read comp_pred flag

            if (s->comppredmode != PRED_SWITCHABLE) {

                b->comp = s->comppredmode == PRED_COMPREF;

            } else {

                int c;



                // FIXME add intra as ref=0xff (or -1) to make these easier?

                if (have_a) {

                    if (have_l) {

                        if (s->above_comp_ctx[col] && s->left_comp_ctx[row7]) {

                            c = 4;

                        } else if (s->above_comp_ctx[col]) {

                            c = 2 + (s->left_intra_ctx[row7] ||

                                     s->left_ref_ctx[row7] == s->fixcompref);

                        } else if (s->left_comp_ctx[row7]) {

                            c = 2 + (s->above_intra_ctx[col] ||

                                     s->above_ref_ctx[col] == s->fixcompref);

                        } else {

                            c = (!s->above_intra_ctx[col] &&

                                 s->above_ref_ctx[col] == s->fixcompref) ^

                            (!s->left_intra_ctx[row7] &&

                             s->left_ref_ctx[row & 7] == s->fixcompref);

                        }

                    } else {

                        c = s->above_comp_ctx[col] ? 3 :

                        (!s->above_intra_ctx[col] && s->above_ref_ctx[col] == s->fixcompref);

                    }

                } else if (have_l) {

                    c = s->left_comp_ctx[row7] ? 3 :

                    (!s->left_intra_ctx[row7] && s->left_ref_ctx[row7] == s->fixcompref);

                } else {

                    c = 1;

                }

                b->comp = vp56_rac_get_prob(&s->c, s->prob.p.comp[c]);

                s->counts.comp[c][b->comp]++;

            }



            // read actual references

            // FIXME probably cache a few variables here to prevent repetitive

            // memory accesses below

            if (b->comp) /* two references */ {

                int fix_idx = s->signbias[s->fixcompref], var_idx = !fix_idx, c, bit;



                b->ref[fix_idx] = s->fixcompref;

                // FIXME can this codeblob be replaced by some sort of LUT?

                if (have_a) {

                    if (have_l) {

                        if (s->above_intra_ctx[col]) {

                            if (s->left_intra_ctx[row7]) {

                                c = 2;

                            } else {

                                c = 1 + 2 * (s->left_ref_ctx[row7] != s->varcompref[1]);

                            }

                        } else if (s->left_intra_ctx[row7]) {

                            c = 1 + 2 * (s->above_ref_ctx[col] != s->varcompref[1]);

                        } else {

                            int refl = s->left_ref_ctx[row7], refa = s->above_ref_ctx[col];



                            if (refl == refa && refa == s->varcompref[1]) {

                                c = 0;

                            } else if (!s->left_comp_ctx[row7] && !s->above_comp_ctx[col]) {

                                if ((refa == s->fixcompref && refl == s->varcompref[0]) ||

                                    (refl == s->fixcompref && refa == s->varcompref[0])) {

                                    c = 4;

                                } else {

                                    c = (refa == refl) ? 3 : 1;

                                }

                            } else if (!s->left_comp_ctx[row7]) {

                                if (refa == s->varcompref[1] && refl != s->varcompref[1]) {

                                    c = 1;

                                } else {

                                    c = (refl == s->varcompref[1] &&

                                         refa != s->varcompref[1]) ? 2 : 4;

                                }

                            } else if (!s->above_comp_ctx[col]) {

                                if (refl == s->varcompref[1] && refa != s->varcompref[1]) {

                                    c = 1;

                                } else {

                                    c = (refa == s->varcompref[1] &&

                                         refl != s->varcompref[1]) ? 2 : 4;

                                }

                            } else {

                                c = (refl == refa) ? 4 : 2;

                            }

                        }

                    } else {

                        if (s->above_intra_ctx[col]) {

                            c = 2;

                        } else if (s->above_comp_ctx[col]) {

                            c = 4 * (s->above_ref_ctx[col] != s->varcompref[1]);

                        } else {

                            c = 3 * (s->above_ref_ctx[col] != s->varcompref[1]);

                        }

                    }

                } else if (have_l) {

                    if (s->left_intra_ctx[row7]) {

                        c = 2;

                    } else if (s->left_comp_ctx[row7]) {

                        c = 4 * (s->left_ref_ctx[row7] != s->varcompref[1]);

                    } else {

                        c = 3 * (s->left_ref_ctx[row7] != s->varcompref[1]);

                    }

                } else {

                    c = 2;

                }

                bit = vp56_rac_get_prob(&s->c, s->prob.p.comp_ref[c]);

                b->ref[var_idx] = s->varcompref[bit];

                s->counts.comp_ref[c][bit]++;

            } else /* single reference */ {

                int bit, c;



                if (have_a && !s->above_intra_ctx[col]) {

                    if (have_l && !s->left_intra_ctx[row7]) {

                        if (s->left_comp_ctx[row7]) {

                            if (s->above_comp_ctx[col]) {

                                c = 1 + (!s->fixcompref || !s->left_ref_ctx[row7] ||

                                         !s->above_ref_ctx[col]);

                            } else {

                                c = (3 * !s->above_ref_ctx[col]) +

                                    (!s->fixcompref || !s->left_ref_ctx[row7]);

                            }

                        } else if (s->above_comp_ctx[col]) {

                            c = (3 * !s->left_ref_ctx[row7]) +

                                (!s->fixcompref || !s->above_ref_ctx[col]);

                        } else {

                            c = 2 * !s->left_ref_ctx[row7] + 2 * !s->above_ref_ctx[col];

                        }

                    } else if (s->above_intra_ctx[col]) {

                        c = 2;

                    } else if (s->above_comp_ctx[col]) {

                        c = 1 + (!s->fixcompref || !s->above_ref_ctx[col]);

                    } else {

                        c = 4 * (!s->above_ref_ctx[col]);

                    }

                } else if (have_l && !s->left_intra_ctx[row7]) {

                    if (s->left_intra_ctx[row7]) {

                        c = 2;

                    } else if (s->left_comp_ctx[row7]) {

                        c = 1 + (!s->fixcompref || !s->left_ref_ctx[row7]);

                    } else {

                        c = 4 * (!s->left_ref_ctx[row7]);

                    }

                } else {

                    c = 2;

                }

                bit = vp56_rac_get_prob(&s->c, s->prob.p.single_ref[c][0]);

                s->counts.single_ref[c][0][bit]++;

                if (!bit) {

                    b->ref[0] = 0;

                } else {

                    // FIXME can this codeblob be replaced by some sort of LUT?

                    if (have_a) {

                        if (have_l) {

                            if (s->left_intra_ctx[row7]) {

                                if (s->above_intra_ctx[col]) {

                                    c = 2;

                                } else if (s->above_comp_ctx[col]) {

                                    c = 1 + 2 * (s->fixcompref == 1 ||

                                                 s->above_ref_ctx[col] == 1);

                                } else if (!s->above_ref_ctx[col]) {

                                    c = 3;

                                } else {

                                    c = 4 * (s->above_ref_ctx[col] == 1);

                                }

                            } else if (s->above_intra_ctx[col]) {

                                if (s->left_intra_ctx[row7]) {

                                    c = 2;

                                } else if (s->left_comp_ctx[row7]) {

                                    c = 1 + 2 * (s->fixcompref == 1 ||

                                                 s->left_ref_ctx[row7] == 1);

                                } else if (!s->left_ref_ctx[row7]) {

                                    c = 3;

                                } else {

                                    c = 4 * (s->left_ref_ctx[row7] == 1);

                                }

                            } else if (s->above_comp_ctx[col]) {

                                if (s->left_comp_ctx[row7]) {

                                    if (s->left_ref_ctx[row7] == s->above_ref_ctx[col]) {

                                        c = 3 * (s->fixcompref == 1 ||

                                                 s->left_ref_ctx[row7] == 1);

                                    } else {

                                        c = 2;

                                    }

                                } else if (!s->left_ref_ctx[row7]) {

                                    c = 1 + 2 * (s->fixcompref == 1 ||

                                                 s->above_ref_ctx[col] == 1);

                                } else {

                                    c = 3 * (s->left_ref_ctx[row7] == 1) +

                                    (s->fixcompref == 1 || s->above_ref_ctx[col] == 1);

                                }

                            } else if (s->left_comp_ctx[row7]) {

                                if (!s->above_ref_ctx[col]) {

                                    c = 1 + 2 * (s->fixcompref == 1 ||

                                                 s->left_ref_ctx[row7] == 1);

                                } else {

                                    c = 3 * (s->above_ref_ctx[col] == 1) +

                                    (s->fixcompref == 1 || s->left_ref_ctx[row7] == 1);

                                }

                            } else if (!s->above_ref_ctx[col]) {

                                if (!s->left_ref_ctx[row7]) {

                                    c = 3;

                                } else {

                                    c = 4 * (s->left_ref_ctx[row7] == 1);

                                }

                            } else if (!s->left_ref_ctx[row7]) {

                                c = 4 * (s->above_ref_ctx[col] == 1);

                            } else {

                                c = 2 * (s->left_ref_ctx[row7] == 1) +

                                2 * (s->above_ref_ctx[col] == 1);

                            }

                        } else {

                            if (s->above_intra_ctx[col] ||

                                (!s->above_comp_ctx[col] && !s->above_ref_ctx[col])) {

                                c = 2;

                            } else if (s->above_comp_ctx[col]) {

                                c = 3 * (s->fixcompref == 1 || s->above_ref_ctx[col] == 1);

                            } else {

                                c = 4 * (s->above_ref_ctx[col] == 1);

                            }

                        }

                    } else if (have_l) {

                        if (s->left_intra_ctx[row7] ||

                            (!s->left_comp_ctx[row7] && !s->left_ref_ctx[row7])) {

                            c = 2;

                        } else if (s->left_comp_ctx[row7]) {

                            c = 3 * (s->fixcompref == 1 || s->left_ref_ctx[row7] == 1);

                        } else {

                            c = 4 * (s->left_ref_ctx[row7] == 1);

                        }

                    } else {

                        c = 2;

                    }

                    bit = vp56_rac_get_prob(&s->c, s->prob.p.single_ref[c][1]);

                    s->counts.single_ref[c][1][bit]++;

                    b->ref[0] = 1 + bit;

                }

            }

        }



        if (b->bs <= BS_8x8) {

            if (s->segmentation.feat[b->seg_id].skip_enabled) {

                b->mode[0] = b->mode[1] = b->mode[2] = b->mode[3] = ZEROMV;

            } else {

                static const uint8_t off[10] = {

                    3, 0, 0, 1, 0, 0, 0, 0, 0, 0

                };



                // FIXME this needs to use the LUT tables from find_ref_mvs

                // because not all are -1,0/0,-1

                int c = inter_mode_ctx_lut[s->above_mode_ctx[col + off[b->bs]]]

                                          [s->left_mode_ctx[row7 + off[b->bs]]];



                b->mode[0] = vp8_rac_get_tree(&s->c, vp9_inter_mode_tree,

                                              s->prob.p.mv_mode[c]);

                b->mode[1] = b->mode[2] = b->mode[3] = b->mode[0];

                s->counts.mv_mode[c][b->mode[0] - 10]++;

            }

        }



        if (s->filtermode == FILTER_SWITCHABLE) {

            int c;



            if (have_a && s->above_mode_ctx[col] >= NEARESTMV) {

                if (have_l && s->left_mode_ctx[row7] >= NEARESTMV) {

                    c = s->above_filter_ctx[col] == s->left_filter_ctx[row7] ?

                        s->left_filter_ctx[row7] : 3;

                } else {

                    c = s->above_filter_ctx[col];

                }

            } else if (have_l && s->left_mode_ctx[row7] >= NEARESTMV) {

                c = s->left_filter_ctx[row7];

            } else {

                c = 3;

            }



            b->filter = vp8_rac_get_tree(&s->c, vp9_filter_tree,

                                         s->prob.p.filter[c]);

            s->counts.filter[c][b->filter]++;

        } else {

            b->filter = s->filtermode;

        }



        if (b->bs > BS_8x8) {

            int c = inter_mode_ctx_lut[s->above_mode_ctx[col]][s->left_mode_ctx[row7]];



            b->mode[0] = vp8_rac_get_tree(&s->c, vp9_inter_mode_tree,

                                          s->prob.p.mv_mode[c]);

            s->counts.mv_mode[c][b->mode[0] - 10]++;

            fill_mv(s, b->mv[0], b->mode[0], 0);



            if (b->bs != BS_8x4) {

                b->mode[1] = vp8_rac_get_tree(&s->c, vp9_inter_mode_tree,

                                              s->prob.p.mv_mode[c]);

                s->counts.mv_mode[c][b->mode[1] - 10]++;

                fill_mv(s, b->mv[1], b->mode[1], 1);

            } else {

                b->mode[1] = b->mode[0];

                AV_COPY32(&b->mv[1][0], &b->mv[0][0]);

                AV_COPY32(&b->mv[1][1], &b->mv[0][1]);

            }



            if (b->bs != BS_4x8) {

                b->mode[2] = vp8_rac_get_tree(&s->c, vp9_inter_mode_tree,

                                              s->prob.p.mv_mode[c]);

                s->counts.mv_mode[c][b->mode[2] - 10]++;

                fill_mv(s, b->mv[2], b->mode[2], 2);



                if (b->bs != BS_8x4) {

                    b->mode[3] = vp8_rac_get_tree(&s->c, vp9_inter_mode_tree,

                                                  s->prob.p.mv_mode[c]);

                    s->counts.mv_mode[c][b->mode[3] - 10]++;

                    fill_mv(s, b->mv[3], b->mode[3], 3);

                } else {

                    b->mode[3] = b->mode[2];

                    AV_COPY32(&b->mv[3][0], &b->mv[2][0]);

                    AV_COPY32(&b->mv[3][1], &b->mv[2][1]);

                }

            } else {

                b->mode[2] = b->mode[0];

                AV_COPY32(&b->mv[2][0], &b->mv[0][0]);

                AV_COPY32(&b->mv[2][1], &b->mv[0][1]);

                b->mode[3] = b->mode[1];

                AV_COPY32(&b->mv[3][0], &b->mv[1][0]);

                AV_COPY32(&b->mv[3][1], &b->mv[1][1]);

            }

        } else {

            fill_mv(s, b->mv[0], b->mode[0], -1);

            AV_COPY32(&b->mv[1][0], &b->mv[0][0]);

            AV_COPY32(&b->mv[2][0], &b->mv[0][0]);

            AV_COPY32(&b->mv[3][0], &b->mv[0][0]);

            AV_COPY32(&b->mv[1][1], &b->mv[0][1]);

            AV_COPY32(&b->mv[2][1], &b->mv[0][1]);

            AV_COPY32(&b->mv[3][1], &b->mv[0][1]);

        }

    }



    // FIXME this can probably be optimized

    memset(&s->above_skip_ctx[col], b->skip, w4);

    memset(&s->left_skip_ctx[row7], b->skip, h4);

    memset(&s->above_txfm_ctx[col], b->tx, w4);

    memset(&s->left_txfm_ctx[row7], b->tx, h4);

    memset(&s->above_partition_ctx[col], above_ctx[b->bs], w4);

    memset(&s->left_partition_ctx[row7], left_ctx[b->bs], h4);

    if (!s->keyframe && !s->intraonly) {

        memset(&s->above_intra_ctx[col], b->intra, w4);

        memset(&s->left_intra_ctx[row7], b->intra, h4);

        memset(&s->above_comp_ctx[col], b->comp, w4);

        memset(&s->left_comp_ctx[row7], b->comp, h4);

        memset(&s->above_mode_ctx[col], b->mode[3], w4);

        memset(&s->left_mode_ctx[row7], b->mode[3], h4);

        if (s->filtermode == FILTER_SWITCHABLE && !b->intra ) {

            memset(&s->above_filter_ctx[col], b->filter, w4);

            memset(&s->left_filter_ctx[row7], b->filter, h4);

            b->filter = vp9_filter_lut[b->filter];

        }

        if (b->bs > BS_8x8) {

            int mv0 = AV_RN32A(&b->mv[3][0]), mv1 = AV_RN32A(&b->mv[3][1]);



            AV_COPY32(&s->left_mv_ctx[row7 * 2 + 0][0], &b->mv[1][0]);

            AV_COPY32(&s->left_mv_ctx[row7 * 2 + 0][1], &b->mv[1][1]);

            AV_WN32A(&s->left_mv_ctx[row7 * 2 + 1][0], mv0);

            AV_WN32A(&s->left_mv_ctx[row7 * 2 + 1][1], mv1);

            AV_COPY32(&s->above_mv_ctx[col * 2 + 0][0], &b->mv[2][0]);

            AV_COPY32(&s->above_mv_ctx[col * 2 + 0][1], &b->mv[2][1]);

            AV_WN32A(&s->above_mv_ctx[col * 2 + 1][0], mv0);

            AV_WN32A(&s->above_mv_ctx[col * 2 + 1][1], mv1);

        } else {

            int n, mv0 = AV_RN32A(&b->mv[3][0]), mv1 = AV_RN32A(&b->mv[3][1]);



            for (n = 0; n < w4 * 2; n++) {

                AV_WN32A(&s->above_mv_ctx[col * 2 + n][0], mv0);

                AV_WN32A(&s->above_mv_ctx[col * 2 + n][1], mv1);

            }

            for (n = 0; n < h4 * 2; n++) {

                AV_WN32A(&s->left_mv_ctx[row7 * 2 + n][0], mv0);

                AV_WN32A(&s->left_mv_ctx[row7 * 2 + n][1], mv1);

            }

        }



        if (!b->intra) { // FIXME write 0xff or -1 if intra, so we can use this

                         // as a direct check in above branches

            int vref = b->ref[b->comp ? s->signbias[s->varcompref[0]] : 0];



            memset(&s->above_ref_ctx[col], vref, w4);

            memset(&s->left_ref_ctx[row7], vref, h4);

        }

    }



    // FIXME kinda ugly

    for (y = 0; y < h4; y++) {

        int x, o = (row + y) * s->sb_cols * 8 + col;

        struct VP9mvrefPair *mv = &s->frames[CUR_FRAME].mv[o];



        if (b->intra) {

            for (x = 0; x < w4; x++) {

                mv[x].ref[0] =

                mv[x].ref[1] = -1;

            }

        } else if (b->comp) {

            for (x = 0; x < w4; x++) {

                mv[x].ref[0] = b->ref[0];

                mv[x].ref[1] = b->ref[1];

                AV_COPY32(&mv[x].mv[0], &b->mv[3][0]);

                AV_COPY32(&mv[x].mv[1], &b->mv[3][1]);

            }

        } else {

            for (x = 0; x < w4; x++) {

                mv[x].ref[0] = b->ref[0];

                mv[x].ref[1] = -1;

                AV_COPY32(&mv[x].mv[0], &b->mv[3][0]);

            }

        }

    }

}
