static int mjpeg_decode_scan_progressive_ac(MJpegDecodeContext *s, int ss,

                                            int se, int Ah, int Al)

{

    int mb_x, mb_y;

    int EOBRUN = 0;

    int c = s->comp_index[0];

    uint8_t *data = s->picture_ptr->data[c];

    int linesize  = s->linesize[c];

    int last_scan = 0;

    int16_t *quant_matrix = s->quant_matrixes[s->quant_sindex[0]];

    int bytes_per_pixel = 1 + (s->bits > 8);



    av_assert0(ss>=0 && Ah>=0 && Al>=0);

    if (se < ss || se > 63) {

        av_log(s->avctx, AV_LOG_ERROR, "SS/SE %d/%d is invalid\n", ss, se);

        return AVERROR_INVALIDDATA;

    }



    if (!Al) {

        s->coefs_finished[c] |= (2LL << se) - (1LL << ss);

        last_scan = !~s->coefs_finished[c];

    }



    if (s->interlaced && s->bottom_field)

        data += linesize >> 1;



    s->restart_count = 0;



    for (mb_y = 0; mb_y < s->mb_height; mb_y++) {

        uint8_t *ptr     = data + (mb_y * linesize * 8 >> s->avctx->lowres);

        int block_idx    = mb_y * s->block_stride[c];

        int16_t (*block)[64] = &s->blocks[c][block_idx];

        uint8_t *last_nnz    = &s->last_nnz[c][block_idx];

        for (mb_x = 0; mb_x < s->mb_width; mb_x++, block++, last_nnz++) {

                int ret;

                if (s->restart_interval && !s->restart_count)

                    s->restart_count = s->restart_interval;



                if (Ah)

                    ret = decode_block_refinement(s, *block, last_nnz, s->ac_index[0],

                                                  quant_matrix, ss, se, Al, &EOBRUN);

                else

                    ret = decode_block_progressive(s, *block, last_nnz, s->ac_index[0],

                                                   quant_matrix, ss, se, Al, &EOBRUN);

                if (ret < 0) {

                    av_log(s->avctx, AV_LOG_ERROR,

                           "error y=%d x=%d\n", mb_y, mb_x);

                    return AVERROR_INVALIDDATA;

                }



            if (last_scan) {

                    s->dsp.idct_put(ptr, linesize, *block);

                    if (s->bits & 7)

                        shift_output(s, ptr, linesize);

                    ptr += bytes_per_pixel*8 >> s->avctx->lowres;

            }

            if (handle_rstn(s, 0))

                EOBRUN = 0;

        }

    }

    return 0;

}
