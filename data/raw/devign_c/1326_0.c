static av_noinline void FUNC(hl_decode_mb_444)(const H264Context *h, H264SliceContext *sl)

{

    const int mb_x    = sl->mb_x;

    const int mb_y    = sl->mb_y;

    const int mb_xy   = sl->mb_xy;

    const int mb_type = h->cur_pic.mb_type[mb_xy];

    uint8_t *dest[3];

    int linesize;

    int i, j, p;

    const int *block_offset = &h->block_offset[0];

    const int transform_bypass = !SIMPLE && (sl->qscale == 0 && h->sps.transform_bypass);

    const int plane_count      = (SIMPLE || !CONFIG_GRAY || !(h->flags & AV_CODEC_FLAG_GRAY)) ? 3 : 1;



    for (p = 0; p < plane_count; p++) {

        dest[p] = h->cur_pic.f->data[p] +

                  ((mb_x << PIXEL_SHIFT) + mb_y * sl->linesize) * 16;

        h->vdsp.prefetch(dest[p] + (sl->mb_x & 3) * 4 * sl->linesize + (64 << PIXEL_SHIFT),

                         sl->linesize, 4);

    }



    h->list_counts[mb_xy] = sl->list_count;



    if (!SIMPLE && MB_FIELD(sl)) {

        linesize     = sl->mb_linesize = sl->mb_uvlinesize = sl->linesize * 2;

        block_offset = &h->block_offset[48];

        if (mb_y & 1) // FIXME move out of this function?

            for (p = 0; p < 3; p++)

                dest[p] -= sl->linesize * 15;

        if (FRAME_MBAFF(h)) {

            int list;

            for (list = 0; list < sl->list_count; list++) {

                if (!USES_LIST(mb_type, list))

                    continue;

                if (IS_16X16(mb_type)) {

                    int8_t *ref = &sl->ref_cache[list][scan8[0]];

                    fill_rectangle(ref, 4, 4, 8, (16 + *ref) ^ (sl->mb_y & 1), 1);

                } else {

                    for (i = 0; i < 16; i += 4) {

                        int ref = sl->ref_cache[list][scan8[i]];

                        if (ref >= 0)

                            fill_rectangle(&sl->ref_cache[list][scan8[i]], 2, 2,

                                           8, (16 + ref) ^ (sl->mb_y & 1), 1);

                    }

                }

            }

        }

    } else {

        linesize = sl->mb_linesize = sl->mb_uvlinesize = sl->linesize;

    }



    if (!SIMPLE && IS_INTRA_PCM(mb_type)) {

        if (PIXEL_SHIFT) {

            const int bit_depth = h->sps.bit_depth_luma;

            GetBitContext gb;

            init_get_bits(&gb, sl->intra_pcm_ptr, 768 * bit_depth);



            for (p = 0; p < plane_count; p++)

                for (i = 0; i < 16; i++) {

                    uint16_t *tmp = (uint16_t *)(dest[p] + i * linesize);

                    for (j = 0; j < 16; j++)

                        tmp[j] = get_bits(&gb, bit_depth);

                }

        } else {

            for (p = 0; p < plane_count; p++)

                for (i = 0; i < 16; i++)

                    memcpy(dest[p] + i * linesize,

                           sl->intra_pcm_ptr + p * 256 + i * 16, 16);

        }

    } else {

        if (IS_INTRA(mb_type)) {

            if (sl->deblocking_filter)

                xchg_mb_border(h, sl, dest[0], dest[1], dest[2], linesize,

                               linesize, 1, 1, SIMPLE, PIXEL_SHIFT);



            for (p = 0; p < plane_count; p++)

                hl_decode_mb_predict_luma(h, sl, mb_type, SIMPLE,

                                          transform_bypass, PIXEL_SHIFT,

                                          block_offset, linesize, dest[p], p);



            if (sl->deblocking_filter)

                xchg_mb_border(h, sl, dest[0], dest[1], dest[2], linesize,

                               linesize, 0, 1, SIMPLE, PIXEL_SHIFT);

        } else {

            FUNC(hl_motion_444)(h, sl, dest[0], dest[1], dest[2],

                      h->qpel_put, h->h264chroma.put_h264_chroma_pixels_tab,

                      h->qpel_avg, h->h264chroma.avg_h264_chroma_pixels_tab,

                      h->h264dsp.weight_h264_pixels_tab,

                      h->h264dsp.biweight_h264_pixels_tab);

        }



        for (p = 0; p < plane_count; p++)

            hl_decode_mb_idct_luma(h, sl, mb_type, SIMPLE, transform_bypass,

                                   PIXEL_SHIFT, block_offset, linesize,

                                   dest[p], p);

    }

}
