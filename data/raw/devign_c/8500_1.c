static int vp9_decode_frame(AVCodecContext *avctx, AVFrame *frame,

                            int *got_frame, const uint8_t *data, int size)

{

    VP9Context *s = avctx->priv_data;

    int ret, tile_row, tile_col, i, ref = -1, row, col;

    ptrdiff_t yoff = 0, uvoff = 0;



    ret = decode_frame_header(avctx, data, size, &ref);

    if (ret < 0) {

        return ret;

    } else if (!ret) {

        if (!s->refs[ref]->buf[0]) {

            av_log(avctx, AV_LOG_ERROR,

                   "Requested reference %d not available\n", ref);

            return AVERROR_INVALIDDATA;

        }



        ret = av_frame_ref(frame, s->refs[ref]);

        if (ret < 0)

            return ret;

        *got_frame = 1;

        return 0;

    }

    data += ret;

    size -= ret;



    s->cur_frame = frame;



    av_frame_unref(s->cur_frame);

    if ((ret = ff_get_buffer(avctx, s->cur_frame,

                             s->refreshrefmask ? AV_GET_BUFFER_FLAG_REF : 0)) < 0)

        return ret;

    s->cur_frame->key_frame = s->keyframe;

    s->cur_frame->pict_type = s->keyframe ? AV_PICTURE_TYPE_I

                                          : AV_PICTURE_TYPE_P;



    // main tile decode loop

    memset(s->above_partition_ctx, 0, s->cols);

    memset(s->above_skip_ctx, 0, s->cols);

    if (s->keyframe || s->intraonly)

        memset(s->above_mode_ctx, DC_PRED, s->cols * 2);

    else

        memset(s->above_mode_ctx, NEARESTMV, s->cols);

    memset(s->above_y_nnz_ctx, 0, s->sb_cols * 16);

    memset(s->above_uv_nnz_ctx[0], 0, s->sb_cols * 8);

    memset(s->above_uv_nnz_ctx[1], 0, s->sb_cols * 8);

    memset(s->above_segpred_ctx, 0, s->cols);

    for (tile_row = 0; tile_row < s->tiling.tile_rows; tile_row++) {

        set_tile_offset(&s->tiling.tile_row_start, &s->tiling.tile_row_end,

                        tile_row, s->tiling.log2_tile_rows, s->sb_rows);

        for (tile_col = 0; tile_col < s->tiling.tile_cols; tile_col++) {

            int64_t tile_size;



            if (tile_col == s->tiling.tile_cols - 1 &&

                tile_row == s->tiling.tile_rows - 1) {

                tile_size = size;

            } else {

                tile_size = AV_RB32(data);

                data     += 4;

                size     -= 4;

            }

            if (tile_size > size)

                return AVERROR_INVALIDDATA;

            ff_vp56_init_range_decoder(&s->c_b[tile_col], data, tile_size);

            if (vp56_rac_get_prob_branchy(&s->c_b[tile_col], 128)) // marker bit

                return AVERROR_INVALIDDATA;

            data += tile_size;

            size -= tile_size;

        }



        for (row = s->tiling.tile_row_start;

             row < s->tiling.tile_row_end;

             row += 8, yoff += s->cur_frame->linesize[0] * 64,

             uvoff += s->cur_frame->linesize[1] * 32) {

            VP9Filter *lflvl = s->lflvl;

            ptrdiff_t yoff2 = yoff, uvoff2 = uvoff;



            for (tile_col = 0; tile_col < s->tiling.tile_cols; tile_col++) {

                set_tile_offset(&s->tiling.tile_col_start,

                                &s->tiling.tile_col_end,

                                tile_col, s->tiling.log2_tile_cols, s->sb_cols);



                memset(s->left_partition_ctx, 0, 8);

                memset(s->left_skip_ctx, 0, 8);

                if (s->keyframe || s->intraonly)

                    memset(s->left_mode_ctx, DC_PRED, 16);

                else

                    memset(s->left_mode_ctx, NEARESTMV, 8);

                memset(s->left_y_nnz_ctx, 0, 16);

                memset(s->left_uv_nnz_ctx, 0, 16);

                memset(s->left_segpred_ctx, 0, 8);



                memcpy(&s->c, &s->c_b[tile_col], sizeof(s->c));

                for (col = s->tiling.tile_col_start;

                     col < s->tiling.tile_col_end;

                     col += 8, yoff2 += 64, uvoff2 += 32, lflvl++) {

                    // FIXME integrate with lf code (i.e. zero after each

                    // use, similar to invtxfm coefficients, or similar)

                    memset(lflvl->mask, 0, sizeof(lflvl->mask));



                    if ((ret = decode_subblock(avctx, row, col, lflvl,

                                               yoff2, uvoff2, BL_64X64)) < 0)

                        return ret;

                }

                memcpy(&s->c_b[tile_col], &s->c, sizeof(s->c));

            }



            // backup pre-loopfilter reconstruction data for intra

            // prediction of next row of sb64s

            if (row + 8 < s->rows) {

                memcpy(s->intra_pred_data[0],

                       s->cur_frame->data[0] + yoff +

                       63 * s->cur_frame->linesize[0],

                       8 * s->cols);

                memcpy(s->intra_pred_data[1],

                       s->cur_frame->data[1] + uvoff +

                       31 * s->cur_frame->linesize[1],

                       4 * s->cols);

                memcpy(s->intra_pred_data[2],

                       s->cur_frame->data[2] + uvoff +

                       31 * s->cur_frame->linesize[2],

                       4 * s->cols);

            }



            // loopfilter one row

            if (s->filter.level) {

                yoff2  = yoff;

                uvoff2 = uvoff;

                lflvl  = s->lflvl;

                for (col = 0; col < s->cols;

                     col += 8, yoff2 += 64, uvoff2 += 32, lflvl++)

                    loopfilter_subblock(avctx, lflvl, row, col, yoff2, uvoff2);

            }

        }

    }



    // bw adaptivity (or in case of parallel decoding mode, fw adaptivity

    // probability maintenance between frames)

    if (s->refreshctx) {

        if (s->parallelmode) {

            memcpy(s->prob_ctx[s->framectxid].coef, s->prob.coef,

                   sizeof(s->prob.coef));

            s->prob_ctx[s->framectxid].p = s->prob.p;

        } else {

            ff_vp9_adapt_probs(s);

        }

    }

    FFSWAP(VP9MVRefPair *, s->mv[0], s->mv[1]);



    // ref frame setup

    for (i = 0; i < 8; i++)

        if (s->refreshrefmask & (1 << i)) {

            av_frame_unref(s->refs[i]);

            ret = av_frame_ref(s->refs[i], s->cur_frame);

            if (ret < 0)

                return ret;

        }



    if (s->invisible)

        av_frame_unref(s->cur_frame);

    else

        *got_frame = 1;



    return 0;

}
