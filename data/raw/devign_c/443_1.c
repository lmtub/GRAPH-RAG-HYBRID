static int decode_slice(struct AVCodecContext *avctx, void *arg)

{

    H264Context *h = *(void **)arg;

    int lf_x_start = h->mb_x;



    h->mb_skip_run = -1;



    av_assert0(h->block_offset[15] == (4 * ((scan8[15] - scan8[0]) & 7) << h->pixel_shift) + 4 * h->linesize * ((scan8[15] - scan8[0]) >> 3));



    h->is_complex = FRAME_MBAFF(h) || h->picture_structure != PICT_FRAME ||

                    avctx->codec_id != AV_CODEC_ID_H264 ||

                    (CONFIG_GRAY && (h->flags & CODEC_FLAG_GRAY));



    if (!(h->avctx->active_thread_type & FF_THREAD_SLICE) && h->picture_structure == PICT_FRAME) {

        const int start_i  = av_clip(h->resync_mb_x + h->resync_mb_y * h->mb_width, 0, h->mb_num - 1);

        if (start_i) {

            int prev_status = h->er.error_status_table[h->er.mb_index2xy[start_i - 1]];

            prev_status &= ~ VP_START;

            if (prev_status != (ER_MV_END | ER_DC_END | ER_AC_END))

                h->er.error_occurred = 1;

        }

    }



    if (h->pps.cabac) {

        /* realign */

        align_get_bits(&h->gb);



        /* init cabac */

        ff_init_cabac_decoder(&h->cabac,

                              h->gb.buffer + get_bits_count(&h->gb) / 8,

                              (get_bits_left(&h->gb) + 7) / 8);



        ff_h264_init_cabac_states(h);



        for (;;) {

            // START_TIMER

            int ret = ff_h264_decode_mb_cabac(h);

            int eos;

            // STOP_TIMER("decode_mb_cabac")



            if (ret >= 0)

                ff_h264_hl_decode_mb(h);



            // FIXME optimal? or let mb_decode decode 16x32 ?

            if (ret >= 0 && FRAME_MBAFF(h)) {

                h->mb_y++;



                ret = ff_h264_decode_mb_cabac(h);



                if (ret >= 0)

                    ff_h264_hl_decode_mb(h);

                h->mb_y--;

            }

            eos = get_cabac_terminate(&h->cabac);



            if ((h->workaround_bugs & FF_BUG_TRUNCATED) &&

                h->cabac.bytestream > h->cabac.bytestream_end + 2) {

                er_add_slice(h, h->resync_mb_x, h->resync_mb_y, h->mb_x - 1,

                                h->mb_y, ER_MB_END);

                if (h->mb_x >= lf_x_start)

                    loop_filter(h, lf_x_start, h->mb_x + 1);

                return 0;

            }

            if (h->cabac.bytestream > h->cabac.bytestream_end + 2 )

                av_log(h->avctx, AV_LOG_DEBUG, "bytestream overread %td\n", h->cabac.bytestream_end - h->cabac.bytestream);

            if (ret < 0 || h->cabac.bytestream > h->cabac.bytestream_end + 4) {

                av_log(h->avctx, AV_LOG_ERROR,

                       "error while decoding MB %d %d, bytestream (%td)\n",

                       h->mb_x, h->mb_y,

                       h->cabac.bytestream_end - h->cabac.bytestream);

                er_add_slice(h, h->resync_mb_x, h->resync_mb_y, h->mb_x,

                                h->mb_y, ER_MB_ERROR);

                return -1;

            }



            if (++h->mb_x >= h->mb_width) {

                loop_filter(h, lf_x_start, h->mb_x);

                h->mb_x = lf_x_start = 0;

                decode_finish_row(h);

                ++h->mb_y;

                if (FIELD_OR_MBAFF_PICTURE(h)) {

                    ++h->mb_y;

                    if (FRAME_MBAFF(h) && h->mb_y < h->mb_height)

                        predict_field_decoding_flag(h);

                }

            }



            if (eos || h->mb_y >= h->mb_height) {

                tprintf(h->avctx, "slice end %d %d\n",

                        get_bits_count(&h->gb), h->gb.size_in_bits);

                er_add_slice(h, h->resync_mb_x, h->resync_mb_y, h->mb_x - 1,

                                h->mb_y, ER_MB_END);

                if (h->mb_x > lf_x_start)

                    loop_filter(h, lf_x_start, h->mb_x);

                return 0;

            }

        }

    } else {

        for (;;) {

            int ret = ff_h264_decode_mb_cavlc(h);



            if (ret >= 0)

                ff_h264_hl_decode_mb(h);



            // FIXME optimal? or let mb_decode decode 16x32 ?

            if (ret >= 0 && FRAME_MBAFF(h)) {

                h->mb_y++;

                ret = ff_h264_decode_mb_cavlc(h);



                if (ret >= 0)

                    ff_h264_hl_decode_mb(h);

                h->mb_y--;

            }



            if (ret < 0) {

                av_log(h->avctx, AV_LOG_ERROR,

                       "error while decoding MB %d %d\n", h->mb_x, h->mb_y);

                er_add_slice(h, h->resync_mb_x, h->resync_mb_y, h->mb_x,

                                h->mb_y, ER_MB_ERROR);

                return -1;

            }



            if (++h->mb_x >= h->mb_width) {

                loop_filter(h, lf_x_start, h->mb_x);

                h->mb_x = lf_x_start = 0;

                decode_finish_row(h);

                ++h->mb_y;

                if (FIELD_OR_MBAFF_PICTURE(h)) {

                    ++h->mb_y;

                    if (FRAME_MBAFF(h) && h->mb_y < h->mb_height)

                        predict_field_decoding_flag(h);

                }

                if (h->mb_y >= h->mb_height) {

                    tprintf(h->avctx, "slice end %d %d\n",

                            get_bits_count(&h->gb), h->gb.size_in_bits);



                    if (   get_bits_left(&h->gb) == 0

                        || get_bits_left(&h->gb) > 0 && !(h->avctx->err_recognition & AV_EF_AGGRESSIVE)) {

                        er_add_slice(h, h->resync_mb_x, h->resync_mb_y,

                                        h->mb_x - 1, h->mb_y,

                                        ER_MB_END);



                        return 0;

                    } else {

                        er_add_slice(h, h->resync_mb_x, h->resync_mb_y,

                                        h->mb_x, h->mb_y,

                                        ER_MB_END);



                        return -1;

                    }

                }

            }



            if (get_bits_left(&h->gb) <= 0 && h->mb_skip_run <= 0) {

                tprintf(h->avctx, "slice end %d %d\n",

                        get_bits_count(&h->gb), h->gb.size_in_bits);

                if (get_bits_left(&h->gb) == 0) {

                    er_add_slice(h, h->resync_mb_x, h->resync_mb_y,

                                    h->mb_x - 1, h->mb_y,

                                    ER_MB_END);

                    if (h->mb_x > lf_x_start)

                        loop_filter(h, lf_x_start, h->mb_x);



                    return 0;

                } else {

                    er_add_slice(h, h->resync_mb_x, h->resync_mb_y, h->mb_x,

                                    h->mb_y, ER_MB_ERROR);



                    return -1;

                }

            }

        }

    }

}
