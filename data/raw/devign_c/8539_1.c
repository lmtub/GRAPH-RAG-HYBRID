static av_always_inline void decode_mb_row_no_filter(AVCodecContext *avctx, void *tdata,

                                        int jobnr, int threadnr, int is_vp7)

{

    VP8Context *s = avctx->priv_data;

    VP8ThreadData *prev_td, *next_td, *td = &s->thread_data[threadnr];

    int mb_y = td->thread_mb_pos >> 16;

    int mb_x, mb_xy = mb_y * s->mb_width;

    int num_jobs = s->num_jobs;

    VP8Frame *curframe = s->curframe, *prev_frame = s->prev_frame;

    VP56RangeCoder *c  = &s->coeff_partition[mb_y & (s->num_coeff_partitions - 1)];

    VP8Macroblock *mb;

    uint8_t *dst[3] = {

        curframe->tf.f->data[0] + 16 * mb_y * s->linesize,

        curframe->tf.f->data[1] +  8 * mb_y * s->uvlinesize,

        curframe->tf.f->data[2] +  8 * mb_y * s->uvlinesize

    };

    if (mb_y == 0)

        prev_td = td;

    else

        prev_td = &s->thread_data[(jobnr + num_jobs - 1) % num_jobs];

    if (mb_y == s->mb_height - 1)

        next_td = td;

    else

        next_td = &s->thread_data[(jobnr + 1) % num_jobs];

    if (s->mb_layout == 1)

        mb = s->macroblocks_base + ((s->mb_width + 1) * (mb_y + 1) + 1);

    else {

        // Make sure the previous frame has read its segmentation map,

        // if we re-use the same map.

        if (prev_frame && s->segmentation.enabled &&

            !s->segmentation.update_map)

            ff_thread_await_progress(&prev_frame->tf, mb_y, 0);

        mb = s->macroblocks + (s->mb_height - mb_y - 1) * 2;

        memset(mb - 1, 0, sizeof(*mb)); // zero left macroblock

        AV_WN32A(s->intra4x4_pred_mode_left, DC_PRED * 0x01010101);

    }



    if (!is_vp7 || mb_y == 0)

        memset(td->left_nnz, 0, sizeof(td->left_nnz));



    s->mv_min.x = -MARGIN;

    s->mv_max.x = ((s->mb_width - 1) << 6) + MARGIN;



    for (mb_x = 0; mb_x < s->mb_width; mb_x++, mb_xy++, mb++) {

        // Wait for previous thread to read mb_x+2, and reach mb_y-1.

        if (prev_td != td) {

            if (threadnr != 0) {

                check_thread_pos(td, prev_td,

                                 mb_x + (is_vp7 ? 2 : 1),

                                 mb_y - (is_vp7 ? 2 : 1));

            } else {

                check_thread_pos(td, prev_td,

                                 mb_x + (is_vp7 ? 2 : 1) + s->mb_width + 3,

                                 mb_y - (is_vp7 ? 2 : 1));

            }

        }



        s->vdsp.prefetch(dst[0] + (mb_x & 3) * 4 * s->linesize + 64,

                         s->linesize, 4);

        s->vdsp.prefetch(dst[1] + (mb_x & 7) * s->uvlinesize + 64,

                         dst[2] - dst[1], 2);



        if (!s->mb_layout)

            decode_mb_mode(s, mb, mb_x, mb_y, curframe->seg_map->data + mb_xy,

                           prev_frame && prev_frame->seg_map ?

                           prev_frame->seg_map->data + mb_xy : NULL, 0, is_vp7);



        prefetch_motion(s, mb, mb_x, mb_y, mb_xy, VP56_FRAME_PREVIOUS);



        if (!mb->skip)

            decode_mb_coeffs(s, td, c, mb, s->top_nnz[mb_x], td->left_nnz, is_vp7);



        if (mb->mode <= MODE_I4x4)

            intra_predict(s, td, dst, mb, mb_x, mb_y, is_vp7);

        else

            inter_predict(s, td, dst, mb, mb_x, mb_y);



        prefetch_motion(s, mb, mb_x, mb_y, mb_xy, VP56_FRAME_GOLDEN);



        if (!mb->skip) {

            idct_mb(s, td, dst, mb);

        } else {

            AV_ZERO64(td->left_nnz);

            AV_WN64(s->top_nnz[mb_x], 0);   // array of 9, so unaligned



            /* Reset DC block predictors if they would exist

             * if the mb had coefficients */

            if (mb->mode != MODE_I4x4 && mb->mode != VP8_MVMODE_SPLIT) {

                td->left_nnz[8]     = 0;

                s->top_nnz[mb_x][8] = 0;

            }

        }



        if (s->deblock_filter)

            filter_level_for_mb(s, mb, &td->filter_strength[mb_x], is_vp7);



        if (s->deblock_filter && num_jobs != 1 && threadnr == num_jobs - 1) {

            if (s->filter.simple)

                backup_mb_border(s->top_border[mb_x + 1], dst[0],

                                 NULL, NULL, s->linesize, 0, 1);

            else

                backup_mb_border(s->top_border[mb_x + 1], dst[0],

                                 dst[1], dst[2], s->linesize, s->uvlinesize, 0);

        }



        prefetch_motion(s, mb, mb_x, mb_y, mb_xy, VP56_FRAME_GOLDEN2);



        dst[0]      += 16;

        dst[1]      += 8;

        dst[2]      += 8;

        s->mv_min.x -= 64;

        s->mv_max.x -= 64;



        if (mb_x == s->mb_width + 1) {

            update_pos(td, mb_y, s->mb_width + 3);

        } else {

            update_pos(td, mb_y, mb_x);

        }

    }

}
