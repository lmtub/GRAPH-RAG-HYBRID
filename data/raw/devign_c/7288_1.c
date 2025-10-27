static int decode_frame_header(AVCodecContext *ctx,

                               const uint8_t *data, int size, int *ref)

{

    VP9Context *s = ctx->priv_data;

    int c, i, j, k, l, m, n, w, h, max, size2, res, sharp;

    int last_invisible;

    const uint8_t *data2;



    /* general header */

    if ((res = init_get_bits8(&s->gb, data, size)) < 0) {

        av_log(ctx, AV_LOG_ERROR, "Failed to initialize bitstream reader\n");

        return res;

    }

    if (get_bits(&s->gb, 2) != 0x2) { // frame marker

        av_log(ctx, AV_LOG_ERROR, "Invalid frame marker\n");

        return AVERROR_INVALIDDATA;

    }

    s->profile = get_bits1(&s->gb);

    if (get_bits1(&s->gb)) { // reserved bit

        av_log(ctx, AV_LOG_ERROR, "Reserved bit should be zero\n");

        return AVERROR_INVALIDDATA;

    }

    if (get_bits1(&s->gb)) {

        *ref = get_bits(&s->gb, 3);

        return 0;

    }

    s->last_uses_2pass = s->uses_2pass;

    s->last_keyframe  = s->keyframe;

    s->keyframe       = !get_bits1(&s->gb);

    last_invisible    = s->invisible;

    s->invisible      = !get_bits1(&s->gb);

    s->errorres       = get_bits1(&s->gb);

    s->use_last_frame_mvs = !s->errorres && !last_invisible;

    if (s->keyframe) {

        if (get_bits_long(&s->gb, 24) != VP9_SYNCCODE) { // synccode

            av_log(ctx, AV_LOG_ERROR, "Invalid sync code\n");

            return AVERROR_INVALIDDATA;

        }

        s->colorspace = get_bits(&s->gb, 3);

        if (s->colorspace == 7) { // RGB = profile 1

            av_log(ctx, AV_LOG_ERROR, "RGB not supported in profile 0\n");

            return AVERROR_INVALIDDATA;

        }

        s->fullrange  = get_bits1(&s->gb);

        // for profile 1, here follows the subsampling bits

        s->refreshrefmask = 0xff;

        w = get_bits(&s->gb, 16) + 1;

        h = get_bits(&s->gb, 16) + 1;

        if (get_bits1(&s->gb)) // display size

            skip_bits(&s->gb, 32);

    } else {

        s->intraonly  = s->invisible ? get_bits1(&s->gb) : 0;

        s->resetctx   = s->errorres ? 0 : get_bits(&s->gb, 2);

        if (s->intraonly) {

            if (get_bits_long(&s->gb, 24) != VP9_SYNCCODE) { // synccode

                av_log(ctx, AV_LOG_ERROR, "Invalid sync code\n");

                return AVERROR_INVALIDDATA;

            }

            s->refreshrefmask = get_bits(&s->gb, 8);

            w = get_bits(&s->gb, 16) + 1;

            h = get_bits(&s->gb, 16) + 1;

            if (get_bits1(&s->gb)) // display size

                skip_bits(&s->gb, 32);

        } else {

            s->refreshrefmask = get_bits(&s->gb, 8);

            s->refidx[0]      = get_bits(&s->gb, 3);

            s->signbias[0]    = get_bits1(&s->gb);

            s->refidx[1]      = get_bits(&s->gb, 3);

            s->signbias[1]    = get_bits1(&s->gb);

            s->refidx[2]      = get_bits(&s->gb, 3);

            s->signbias[2]    = get_bits1(&s->gb);

            if (!s->refs[s->refidx[0]].f->data[0] ||

                !s->refs[s->refidx[1]].f->data[0] ||

                !s->refs[s->refidx[2]].f->data[0]) {

                av_log(ctx, AV_LOG_ERROR, "Not all references are available\n");

                return AVERROR_INVALIDDATA;

            }

            if (get_bits1(&s->gb)) {

                w = s->refs[s->refidx[0]].f->width;

                h = s->refs[s->refidx[0]].f->height;

            } else if (get_bits1(&s->gb)) {

                w = s->refs[s->refidx[1]].f->width;

                h = s->refs[s->refidx[1]].f->height;

            } else if (get_bits1(&s->gb)) {

                w = s->refs[s->refidx[2]].f->width;

                h = s->refs[s->refidx[2]].f->height;

            } else {

                w = get_bits(&s->gb, 16) + 1;

                h = get_bits(&s->gb, 16) + 1;

            }

            // Note that in this code, "CUR_FRAME" is actually before we

            // have formally allocated a frame, and thus actually represents

            // the _last_ frame

            s->use_last_frame_mvs &= s->frames[CUR_FRAME].tf.f->width == w &&

                                     s->frames[CUR_FRAME].tf.f->height == h;

            if (get_bits1(&s->gb)) // display size

                skip_bits(&s->gb, 32);

            s->highprecisionmvs = get_bits1(&s->gb);

            s->filtermode = get_bits1(&s->gb) ? FILTER_SWITCHABLE :

                                                get_bits(&s->gb, 2);

            s->allowcompinter = s->signbias[0] != s->signbias[1] ||

                                s->signbias[0] != s->signbias[2];

            if (s->allowcompinter) {

                if (s->signbias[0] == s->signbias[1]) {

                    s->fixcompref    = 2;

                    s->varcompref[0] = 0;

                    s->varcompref[1] = 1;

                } else if (s->signbias[0] == s->signbias[2]) {

                    s->fixcompref    = 1;

                    s->varcompref[0] = 0;

                    s->varcompref[1] = 2;

                } else {

                    s->fixcompref    = 0;

                    s->varcompref[0] = 1;

                    s->varcompref[1] = 2;

                }

            }

        }

    }

    s->refreshctx   = s->errorres ? 0 : get_bits1(&s->gb);

    s->parallelmode = s->errorres ? 1 : get_bits1(&s->gb);

    s->framectxid   = c = get_bits(&s->gb, 2);



    /* loopfilter header data */

    s->filter.level = get_bits(&s->gb, 6);

    sharp = get_bits(&s->gb, 3);

    // if sharpness changed, reinit lim/mblim LUTs. if it didn't change, keep

    // the old cache values since they are still valid

    if (s->filter.sharpness != sharp)

        memset(s->filter.lim_lut, 0, sizeof(s->filter.lim_lut));

    s->filter.sharpness = sharp;

    if ((s->lf_delta.enabled = get_bits1(&s->gb))) {

        if (get_bits1(&s->gb)) {

            for (i = 0; i < 4; i++)

                if (get_bits1(&s->gb))

                    s->lf_delta.ref[i] = get_sbits_inv(&s->gb, 6);

            for (i = 0; i < 2; i++)

                if (get_bits1(&s->gb))

                    s->lf_delta.mode[i] = get_sbits_inv(&s->gb, 6);

        }

    } else {

        memset(&s->lf_delta, 0, sizeof(s->lf_delta));

    }



    /* quantization header data */

    s->yac_qi      = get_bits(&s->gb, 8);

    s->ydc_qdelta  = get_bits1(&s->gb) ? get_sbits_inv(&s->gb, 4) : 0;

    s->uvdc_qdelta = get_bits1(&s->gb) ? get_sbits_inv(&s->gb, 4) : 0;

    s->uvac_qdelta = get_bits1(&s->gb) ? get_sbits_inv(&s->gb, 4) : 0;

    s->lossless    = s->yac_qi == 0 && s->ydc_qdelta == 0 &&

                     s->uvdc_qdelta == 0 && s->uvac_qdelta == 0;



    /* segmentation header info */

    if ((s->segmentation.enabled = get_bits1(&s->gb))) {

        if ((s->segmentation.update_map = get_bits1(&s->gb))) {

            for (i = 0; i < 7; i++)

                s->prob.seg[i] = get_bits1(&s->gb) ?

                                 get_bits(&s->gb, 8) : 255;

            if ((s->segmentation.temporal = get_bits1(&s->gb)))

                for (i = 0; i < 3; i++)

                    s->prob.segpred[i] = get_bits1(&s->gb) ?

                                         get_bits(&s->gb, 8) : 255;

        } else {

            s->use_last_frame_segmap = !s->keyframe && !s->intraonly &&

                s->frames[CUR_FRAME].tf.f->width == w &&

                s->frames[CUR_FRAME].tf.f->height == h;

        }



        if (get_bits1(&s->gb)) {

            s->segmentation.absolute_vals = get_bits1(&s->gb);

            for (i = 0; i < 8; i++) {

                if ((s->segmentation.feat[i].q_enabled = get_bits1(&s->gb)))

                    s->segmentation.feat[i].q_val = get_sbits_inv(&s->gb, 8);

                if ((s->segmentation.feat[i].lf_enabled = get_bits1(&s->gb)))

                    s->segmentation.feat[i].lf_val = get_sbits_inv(&s->gb, 6);

                if ((s->segmentation.feat[i].ref_enabled = get_bits1(&s->gb)))

                    s->segmentation.feat[i].ref_val = get_bits(&s->gb, 2);

                s->segmentation.feat[i].skip_enabled = get_bits1(&s->gb);

            }

        }

    } else {

        s->segmentation.feat[0].q_enabled    = 0;

        s->segmentation.feat[0].lf_enabled   = 0;

        s->segmentation.feat[0].skip_enabled = 0;

        s->segmentation.feat[0].ref_enabled  = 0;

    }



    // set qmul[] based on Y/UV, AC/DC and segmentation Q idx deltas

    for (i = 0; i < (s->segmentation.enabled ? 8 : 1); i++) {

        int qyac, qydc, quvac, quvdc, lflvl, sh;



        if (s->segmentation.feat[i].q_enabled) {

            if (s->segmentation.absolute_vals)

                qyac = s->segmentation.feat[i].q_val;

            else

                qyac = s->yac_qi + s->segmentation.feat[i].q_val;

        } else {

            qyac  = s->yac_qi;

        }

        qydc  = av_clip_uintp2(qyac + s->ydc_qdelta, 8);

        quvdc = av_clip_uintp2(qyac + s->uvdc_qdelta, 8);

        quvac = av_clip_uintp2(qyac + s->uvac_qdelta, 8);

        qyac  = av_clip_uintp2(qyac, 8);



        s->segmentation.feat[i].qmul[0][0] = vp9_dc_qlookup[qydc];

        s->segmentation.feat[i].qmul[0][1] = vp9_ac_qlookup[qyac];

        s->segmentation.feat[i].qmul[1][0] = vp9_dc_qlookup[quvdc];

        s->segmentation.feat[i].qmul[1][1] = vp9_ac_qlookup[quvac];



        sh = s->filter.level >= 32;

        if (s->segmentation.feat[i].lf_enabled) {

            if (s->segmentation.absolute_vals)

                lflvl = s->segmentation.feat[i].lf_val;

            else

                lflvl = s->filter.level + s->segmentation.feat[i].lf_val;

        } else {

            lflvl  = s->filter.level;

        }

        s->segmentation.feat[i].lflvl[0][0] =

        s->segmentation.feat[i].lflvl[0][1] =

            av_clip_uintp2(lflvl + (s->lf_delta.ref[0] << sh), 6);

        for (j = 1; j < 4; j++) {

            s->segmentation.feat[i].lflvl[j][0] =

                av_clip_uintp2(lflvl + ((s->lf_delta.ref[j] +

                                         s->lf_delta.mode[0]) << sh), 6);

            s->segmentation.feat[i].lflvl[j][1] =

                av_clip_uintp2(lflvl + ((s->lf_delta.ref[j] +

                                         s->lf_delta.mode[1]) << sh), 6);

        }

    }



    /* tiling info */

    if ((res = update_size(ctx, w, h)) < 0) {

        av_log(ctx, AV_LOG_ERROR, "Failed to initialize decoder for %dx%d\n", w, h);

        return res;

    }

    for (s->tiling.log2_tile_cols = 0;

         (s->sb_cols >> s->tiling.log2_tile_cols) > 64;

         s->tiling.log2_tile_cols++) ;

    for (max = 0; (s->sb_cols >> max) >= 4; max++) ;

    max = FFMAX(0, max - 1);

    while (max > s->tiling.log2_tile_cols) {

        if (get_bits1(&s->gb))

            s->tiling.log2_tile_cols++;

        else

            break;

    }

    s->tiling.log2_tile_rows = decode012(&s->gb);

    s->tiling.tile_rows = 1 << s->tiling.log2_tile_rows;

    if (s->tiling.tile_cols != (1 << s->tiling.log2_tile_cols)) {

        s->tiling.tile_cols = 1 << s->tiling.log2_tile_cols;

        s->c_b = av_fast_realloc(s->c_b, &s->c_b_size,

                                 sizeof(VP56RangeCoder) * s->tiling.tile_cols);

        if (!s->c_b) {

            av_log(ctx, AV_LOG_ERROR, "Ran out of memory during range coder init\n");

            return AVERROR(ENOMEM);

        }

    }



    if (s->keyframe || s->errorres || s->intraonly) {

        s->prob_ctx[0].p = s->prob_ctx[1].p = s->prob_ctx[2].p =

                           s->prob_ctx[3].p = vp9_default_probs;

        memcpy(s->prob_ctx[0].coef, vp9_default_coef_probs,

               sizeof(vp9_default_coef_probs));

        memcpy(s->prob_ctx[1].coef, vp9_default_coef_probs,

               sizeof(vp9_default_coef_probs));

        memcpy(s->prob_ctx[2].coef, vp9_default_coef_probs,

               sizeof(vp9_default_coef_probs));

        memcpy(s->prob_ctx[3].coef, vp9_default_coef_probs,

               sizeof(vp9_default_coef_probs));

    }



    // next 16 bits is size of the rest of the header (arith-coded)

    size2 = get_bits(&s->gb, 16);

    data2 = align_get_bits(&s->gb);

    if (size2 > size - (data2 - data)) {

        av_log(ctx, AV_LOG_ERROR, "Invalid compressed header size\n");

        return AVERROR_INVALIDDATA;

    }

    ff_vp56_init_range_decoder(&s->c, data2, size2);

    if (vp56_rac_get_prob_branchy(&s->c, 128)) { // marker bit

        av_log(ctx, AV_LOG_ERROR, "Marker bit was set\n");

        return AVERROR_INVALIDDATA;

    }



    if (s->keyframe || s->intraonly) {

        memset(s->counts.coef, 0, sizeof(s->counts.coef) + sizeof(s->counts.eob));

    } else {

        memset(&s->counts, 0, sizeof(s->counts));

    }

    // FIXME is it faster to not copy here, but do it down in the fw updates

    // as explicit copies if the fw update is missing (and skip the copy upon

    // fw update)?

    s->prob.p = s->prob_ctx[c].p;



    // txfm updates

    if (s->lossless) {

        s->txfmmode = TX_4X4;

    } else {

        s->txfmmode = vp8_rac_get_uint(&s->c, 2);

        if (s->txfmmode == 3)

            s->txfmmode += vp8_rac_get(&s->c);



        if (s->txfmmode == TX_SWITCHABLE) {

            for (i = 0; i < 2; i++)

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.tx8p[i] = update_prob(&s->c, s->prob.p.tx8p[i]);

            for (i = 0; i < 2; i++)

                for (j = 0; j < 2; j++)

                    if (vp56_rac_get_prob_branchy(&s->c, 252))

                        s->prob.p.tx16p[i][j] =

                            update_prob(&s->c, s->prob.p.tx16p[i][j]);

            for (i = 0; i < 2; i++)

                for (j = 0; j < 3; j++)

                    if (vp56_rac_get_prob_branchy(&s->c, 252))

                        s->prob.p.tx32p[i][j] =

                            update_prob(&s->c, s->prob.p.tx32p[i][j]);

        }

    }



    // coef updates

    for (i = 0; i < 4; i++) {

        uint8_t (*ref)[2][6][6][3] = s->prob_ctx[c].coef[i];

        if (vp8_rac_get(&s->c)) {

            for (j = 0; j < 2; j++)

                for (k = 0; k < 2; k++)

                    for (l = 0; l < 6; l++)

                        for (m = 0; m < 6; m++) {

                            uint8_t *p = s->prob.coef[i][j][k][l][m];

                            uint8_t *r = ref[j][k][l][m];

                            if (m >= 3 && l == 0) // dc only has 3 pt

                                break;

                            for (n = 0; n < 3; n++) {

                                if (vp56_rac_get_prob_branchy(&s->c, 252)) {

                                    p[n] = update_prob(&s->c, r[n]);

                                } else {

                                    p[n] = r[n];

                                }

                            }

                            p[3] = 0;

                        }

        } else {

            for (j = 0; j < 2; j++)

                for (k = 0; k < 2; k++)

                    for (l = 0; l < 6; l++)

                        for (m = 0; m < 6; m++) {

                            uint8_t *p = s->prob.coef[i][j][k][l][m];

                            uint8_t *r = ref[j][k][l][m];

                            if (m > 3 && l == 0) // dc only has 3 pt

                                break;

                            memcpy(p, r, 3);

                            p[3] = 0;

                        }

        }

        if (s->txfmmode == i)

            break;

    }



    // mode updates

    for (i = 0; i < 3; i++)

        if (vp56_rac_get_prob_branchy(&s->c, 252))

            s->prob.p.skip[i] = update_prob(&s->c, s->prob.p.skip[i]);

    if (!s->keyframe && !s->intraonly) {

        for (i = 0; i < 7; i++)

            for (j = 0; j < 3; j++)

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.mv_mode[i][j] =

                        update_prob(&s->c, s->prob.p.mv_mode[i][j]);



        if (s->filtermode == FILTER_SWITCHABLE)

            for (i = 0; i < 4; i++)

                for (j = 0; j < 2; j++)

                    if (vp56_rac_get_prob_branchy(&s->c, 252))

                        s->prob.p.filter[i][j] =

                            update_prob(&s->c, s->prob.p.filter[i][j]);



        for (i = 0; i < 4; i++)

            if (vp56_rac_get_prob_branchy(&s->c, 252))

                s->prob.p.intra[i] = update_prob(&s->c, s->prob.p.intra[i]);



        if (s->allowcompinter) {

            s->comppredmode = vp8_rac_get(&s->c);

            if (s->comppredmode)

                s->comppredmode += vp8_rac_get(&s->c);

            if (s->comppredmode == PRED_SWITCHABLE)

                for (i = 0; i < 5; i++)

                    if (vp56_rac_get_prob_branchy(&s->c, 252))

                        s->prob.p.comp[i] =

                            update_prob(&s->c, s->prob.p.comp[i]);

        } else {

            s->comppredmode = PRED_SINGLEREF;

        }



        if (s->comppredmode != PRED_COMPREF) {

            for (i = 0; i < 5; i++) {

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.single_ref[i][0] =

                        update_prob(&s->c, s->prob.p.single_ref[i][0]);

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.single_ref[i][1] =

                        update_prob(&s->c, s->prob.p.single_ref[i][1]);

            }

        }



        if (s->comppredmode != PRED_SINGLEREF) {

            for (i = 0; i < 5; i++)

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.comp_ref[i] =

                        update_prob(&s->c, s->prob.p.comp_ref[i]);

        }



        for (i = 0; i < 4; i++)

            for (j = 0; j < 9; j++)

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.y_mode[i][j] =

                        update_prob(&s->c, s->prob.p.y_mode[i][j]);



        for (i = 0; i < 4; i++)

            for (j = 0; j < 4; j++)

                for (k = 0; k < 3; k++)

                    if (vp56_rac_get_prob_branchy(&s->c, 252))

                        s->prob.p.partition[3 - i][j][k] =

                            update_prob(&s->c, s->prob.p.partition[3 - i][j][k]);



        // mv fields don't use the update_prob subexp model for some reason

        for (i = 0; i < 3; i++)

            if (vp56_rac_get_prob_branchy(&s->c, 252))

                s->prob.p.mv_joint[i] = (vp8_rac_get_uint(&s->c, 7) << 1) | 1;



        for (i = 0; i < 2; i++) {

            if (vp56_rac_get_prob_branchy(&s->c, 252))

                s->prob.p.mv_comp[i].sign = (vp8_rac_get_uint(&s->c, 7) << 1) | 1;



            for (j = 0; j < 10; j++)

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.mv_comp[i].classes[j] =

                        (vp8_rac_get_uint(&s->c, 7) << 1) | 1;



            if (vp56_rac_get_prob_branchy(&s->c, 252))

                s->prob.p.mv_comp[i].class0 = (vp8_rac_get_uint(&s->c, 7) << 1) | 1;



            for (j = 0; j < 10; j++)

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.mv_comp[i].bits[j] =

                        (vp8_rac_get_uint(&s->c, 7) << 1) | 1;

        }



        for (i = 0; i < 2; i++) {

            for (j = 0; j < 2; j++)

                for (k = 0; k < 3; k++)

                    if (vp56_rac_get_prob_branchy(&s->c, 252))

                        s->prob.p.mv_comp[i].class0_fp[j][k] =

                            (vp8_rac_get_uint(&s->c, 7) << 1) | 1;



            for (j = 0; j < 3; j++)

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.mv_comp[i].fp[j] =

                        (vp8_rac_get_uint(&s->c, 7) << 1) | 1;

        }



        if (s->highprecisionmvs) {

            for (i = 0; i < 2; i++) {

                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.mv_comp[i].class0_hp =

                        (vp8_rac_get_uint(&s->c, 7) << 1) | 1;



                if (vp56_rac_get_prob_branchy(&s->c, 252))

                    s->prob.p.mv_comp[i].hp =

                        (vp8_rac_get_uint(&s->c, 7) << 1) | 1;

            }

        }

    }



    return (data2 - data) + size2;

}
