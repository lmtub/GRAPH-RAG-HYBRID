static void inter_recon(AVCodecContext *ctx)

{

    static const uint8_t bwlog_tab[2][N_BS_SIZES] = {

        { 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4 },

        { 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4 },

    };

    VP9Context *s = ctx->priv_data;

    VP9Block *b = s->b;

    int row = s->row, col = s->col;

    ThreadFrame *tref1 = &s->refs[s->refidx[b->ref[0]]];

    AVFrame *ref1 = tref1->f;

    ThreadFrame *tref2 = b->comp ? &s->refs[s->refidx[b->ref[1]]] : NULL;

    AVFrame *ref2 = b->comp ? tref2->f : NULL;

    int w = ctx->width, h = ctx->height;

    ptrdiff_t ls_y = s->y_stride, ls_uv = s->uv_stride;



    // y inter pred

    if (b->bs > BS_8x8) {

        if (b->bs == BS_8x4) {

            mc_luma_dir(s, s->dsp.mc[3][b->filter][0], s->dst[0], ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        row << 3, col << 3, &b->mv[0][0], 8, 4, w, h);

            mc_luma_dir(s, s->dsp.mc[3][b->filter][0],

                        s->dst[0] + 4 * ls_y, ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        (row << 3) + 4, col << 3, &b->mv[2][0], 8, 4, w, h);



            if (b->comp) {

                mc_luma_dir(s, s->dsp.mc[3][b->filter][1], s->dst[0], ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            row << 3, col << 3, &b->mv[0][1], 8, 4, w, h);

                mc_luma_dir(s, s->dsp.mc[3][b->filter][1],

                            s->dst[0] + 4 * ls_y, ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            (row << 3) + 4, col << 3, &b->mv[2][1], 8, 4, w, h);

            }

        } else if (b->bs == BS_4x8) {

            mc_luma_dir(s, s->dsp.mc[4][b->filter][0], s->dst[0], ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        row << 3, col << 3, &b->mv[0][0], 4, 8, w, h);

            mc_luma_dir(s, s->dsp.mc[4][b->filter][0], s->dst[0] + 4, ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        row << 3, (col << 3) + 4, &b->mv[1][0], 4, 8, w, h);



            if (b->comp) {

                mc_luma_dir(s, s->dsp.mc[4][b->filter][1], s->dst[0], ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            row << 3, col << 3, &b->mv[0][1], 4, 8, w, h);

                mc_luma_dir(s, s->dsp.mc[4][b->filter][1], s->dst[0] + 4, ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            row << 3, (col << 3) + 4, &b->mv[1][1], 4, 8, w, h);

            }

        } else {

            av_assert2(b->bs == BS_4x4);



            // FIXME if two horizontally adjacent blocks have the same MV,

            // do a w8 instead of a w4 call

            mc_luma_dir(s, s->dsp.mc[4][b->filter][0], s->dst[0], ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        row << 3, col << 3, &b->mv[0][0], 4, 4, w, h);

            mc_luma_dir(s, s->dsp.mc[4][b->filter][0], s->dst[0] + 4, ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        row << 3, (col << 3) + 4, &b->mv[1][0], 4, 4, w, h);

            mc_luma_dir(s, s->dsp.mc[4][b->filter][0],

                        s->dst[0] + 4 * ls_y, ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        (row << 3) + 4, col << 3, &b->mv[2][0], 4, 4, w, h);

            mc_luma_dir(s, s->dsp.mc[4][b->filter][0],

                        s->dst[0] + 4 * ls_y + 4, ls_y,

                        ref1->data[0], ref1->linesize[0], tref1,

                        (row << 3) + 4, (col << 3) + 4, &b->mv[3][0], 4, 4, w, h);



            if (b->comp) {

                mc_luma_dir(s, s->dsp.mc[4][b->filter][1], s->dst[0], ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            row << 3, col << 3, &b->mv[0][1], 4, 4, w, h);

                mc_luma_dir(s, s->dsp.mc[4][b->filter][1], s->dst[0] + 4, ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            row << 3, (col << 3) + 4, &b->mv[1][1], 4, 4, w, h);

                mc_luma_dir(s, s->dsp.mc[4][b->filter][1],

                            s->dst[0] + 4 * ls_y, ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            (row << 3) + 4, col << 3, &b->mv[2][1], 4, 4, w, h);

                mc_luma_dir(s, s->dsp.mc[4][b->filter][1],

                            s->dst[0] + 4 * ls_y + 4, ls_y,

                            ref2->data[0], ref2->linesize[0], tref2,

                            (row << 3) + 4, (col << 3) + 4, &b->mv[3][1], 4, 4, w, h);

            }

        }

    } else {

        int bwl = bwlog_tab[0][b->bs];

        int bw = bwh_tab[0][b->bs][0] * 4, bh = bwh_tab[0][b->bs][1] * 4;



        mc_luma_dir(s, s->dsp.mc[bwl][b->filter][0], s->dst[0], ls_y,

                    ref1->data[0], ref1->linesize[0], tref1,

                    row << 3, col << 3, &b->mv[0][0],bw, bh, w, h);



        if (b->comp)

            mc_luma_dir(s, s->dsp.mc[bwl][b->filter][1], s->dst[0], ls_y,

                        ref2->data[0], ref2->linesize[0], tref2,

                        row << 3, col << 3, &b->mv[0][1], bw, bh, w, h);

    }



    // uv inter pred

    {

        int bwl = bwlog_tab[1][b->bs];

        int bw = bwh_tab[1][b->bs][0] * 4, bh = bwh_tab[1][b->bs][1] * 4;

        VP56mv mvuv;



        w = (w + 1) >> 1;

        h = (h + 1) >> 1;

        if (b->bs > BS_8x8) {

            mvuv.x = ROUNDED_DIV(b->mv[0][0].x + b->mv[1][0].x + b->mv[2][0].x + b->mv[3][0].x, 4);

            mvuv.y = ROUNDED_DIV(b->mv[0][0].y + b->mv[1][0].y + b->mv[2][0].y + b->mv[3][0].y, 4);

        } else {

            mvuv = b->mv[0][0];

        }



        mc_chroma_dir(s, s->dsp.mc[bwl][b->filter][0],

                      s->dst[1], s->dst[2], ls_uv,

                      ref1->data[1], ref1->linesize[1],

                      ref1->data[2], ref1->linesize[2], tref1,

                      row << 2, col << 2, &mvuv, bw, bh, w, h);



        if (b->comp) {

            if (b->bs > BS_8x8) {

                mvuv.x = ROUNDED_DIV(b->mv[0][1].x + b->mv[1][1].x + b->mv[2][1].x + b->mv[3][1].x, 4);

                mvuv.y = ROUNDED_DIV(b->mv[0][1].y + b->mv[1][1].y + b->mv[2][1].y + b->mv[3][1].y, 4);

            } else {

                mvuv = b->mv[0][1];

            }

            mc_chroma_dir(s, s->dsp.mc[bwl][b->filter][1],

                          s->dst[1], s->dst[2], ls_uv,

                          ref2->data[1], ref2->linesize[1],

                          ref2->data[2], ref2->linesize[2], tref2,

                          row << 2, col << 2, &mvuv, bw, bh, w, h);

        }

    }



    if (!b->skip) {

        /* mostly copied intra_reconn() */



        int w4 = bwh_tab[1][b->bs][0] << 1, step1d = 1 << b->tx, n;

        int h4 = bwh_tab[1][b->bs][1] << 1, x, y, step = 1 << (b->tx * 2);

        int end_x = FFMIN(2 * (s->cols - col), w4);

        int end_y = FFMIN(2 * (s->rows - row), h4);

        int tx = 4 * s->lossless + b->tx, uvtx = b->uvtx + 4 * s->lossless;

        int uvstep1d = 1 << b->uvtx, p;

        uint8_t *dst = s->dst[0];



        // y itxfm add

        for (n = 0, y = 0; y < end_y; y += step1d) {

            uint8_t *ptr = dst;

            for (x = 0; x < end_x; x += step1d, ptr += 4 * step1d, n += step) {

                int eob = b->tx > TX_8X8 ? AV_RN16A(&s->eob[n]) : s->eob[n];



                if (eob)

                    s->dsp.itxfm_add[tx][DCT_DCT](ptr, s->y_stride,

                                                  s->block + 16 * n, eob);

            }

            dst += 4 * s->y_stride * step1d;

        }



        // uv itxfm add

        h4 >>= 1;

        w4 >>= 1;

        end_x >>= 1;

        end_y >>= 1;

        step = 1 << (b->uvtx * 2);

        for (p = 0; p < 2; p++) {

            dst = s->dst[p + 1];

            for (n = 0, y = 0; y < end_y; y += uvstep1d) {

                uint8_t *ptr = dst;

                for (x = 0; x < end_x; x += uvstep1d, ptr += 4 * uvstep1d, n += step) {

                    int eob = b->uvtx > TX_8X8 ? AV_RN16A(&s->uveob[p][n]) : s->uveob[p][n];



                    if (eob)

                        s->dsp.itxfm_add[uvtx][DCT_DCT](ptr, s->uv_stride,

                                                        s->uvblock[p] + 16 * n, eob);

                }

                dst += 4 * uvstep1d * s->uv_stride;

            }

        }

    }

}
