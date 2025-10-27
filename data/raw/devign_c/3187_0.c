static av_always_inline void hl_decode_mb_predict_luma(const H264Context *h,

                                                       H264SliceContext *sl,

                                                       int mb_type, int simple,

                                                       int transform_bypass,

                                                       int pixel_shift,

                                                       const int *block_offset,

                                                       int linesize,

                                                       uint8_t *dest_y, int p)

{

    void (*idct_add)(uint8_t *dst, int16_t *block, int stride);

    void (*idct_dc_add)(uint8_t *dst, int16_t *block, int stride);

    int i;

    int qscale = p == 0 ? sl->qscale : sl->chroma_qp[p - 1];

    block_offset += 16 * p;

    if (IS_INTRA4x4(mb_type)) {

        if (IS_8x8DCT(mb_type)) {

            if (transform_bypass) {

                idct_dc_add =

                idct_add    = h->h264dsp.h264_add_pixels8_clear;

            } else {

                idct_dc_add = h->h264dsp.h264_idct8_dc_add;

                idct_add    = h->h264dsp.h264_idct8_add;

            }

            for (i = 0; i < 16; i += 4) {

                uint8_t *const ptr = dest_y + block_offset[i];

                const int dir      = sl->intra4x4_pred_mode_cache[scan8[i]];

                if (transform_bypass && h->sps.profile_idc == 244 && dir <= 1) {

                    h->hpc.pred8x8l_add[dir](ptr, sl->mb + (i * 16 + p * 256 << pixel_shift), linesize);

                } else {

                    const int nnz = sl->non_zero_count_cache[scan8[i + p * 16]];

                    h->hpc.pred8x8l[dir](ptr, (sl->topleft_samples_available << i) & 0x8000,

                                         (sl->topright_samples_available << i) & 0x4000, linesize);

                    if (nnz) {

                        if (nnz == 1 && dctcoef_get(sl->mb, pixel_shift, i * 16 + p * 256))

                            idct_dc_add(ptr, sl->mb + (i * 16 + p * 256 << pixel_shift), linesize);

                        else

                            idct_add(ptr, sl->mb + (i * 16 + p * 256 << pixel_shift), linesize);

                    }

                }

            }

        } else {

            if (transform_bypass) {

                idct_dc_add  =

                idct_add     = h->h264dsp.h264_add_pixels4_clear;

            } else {

                idct_dc_add = h->h264dsp.h264_idct_dc_add;

                idct_add    = h->h264dsp.h264_idct_add;

            }

            for (i = 0; i < 16; i++) {

                uint8_t *const ptr = dest_y + block_offset[i];

                const int dir      = sl->intra4x4_pred_mode_cache[scan8[i]];



                if (transform_bypass && h->sps.profile_idc == 244 && dir <= 1) {

                    h->hpc.pred4x4_add[dir](ptr, sl->mb + (i * 16 + p * 256 << pixel_shift), linesize);

                } else {

                    uint8_t *topright;

                    int nnz, tr;

                    uint64_t tr_high;

                    if (dir == DIAG_DOWN_LEFT_PRED || dir == VERT_LEFT_PRED) {

                        const int topright_avail = (sl->topright_samples_available << i) & 0x8000;

                        assert(sl->mb_y || linesize <= block_offset[i]);

                        if (!topright_avail) {

                            if (pixel_shift) {

                                tr_high  = ((uint16_t *)ptr)[3 - linesize / 2] * 0x0001000100010001ULL;

                                topright = (uint8_t *)&tr_high;

                            } else {

                                tr       = ptr[3 - linesize] * 0x01010101u;

                                topright = (uint8_t *)&tr;

                            }

                        } else

                            topright = ptr + (4 << pixel_shift) - linesize;

                    } else

                        topright = NULL;



                    h->hpc.pred4x4[dir](ptr, topright, linesize);

                    nnz = sl->non_zero_count_cache[scan8[i + p * 16]];

                    if (nnz) {

                        if (nnz == 1 && dctcoef_get(sl->mb, pixel_shift, i * 16 + p * 256))

                            idct_dc_add(ptr, sl->mb + (i * 16 + p * 256 << pixel_shift), linesize);

                        else

                            idct_add(ptr, sl->mb + (i * 16 + p * 256 << pixel_shift), linesize);

                    }

                }

            }

        }

    } else {

        h->hpc.pred16x16[sl->intra16x16_pred_mode](dest_y, linesize);

        if (sl->non_zero_count_cache[scan8[LUMA_DC_BLOCK_INDEX + p]]) {

            if (!transform_bypass)

                h->h264dsp.h264_luma_dc_dequant_idct(sl->mb + (p * 256 << pixel_shift),

                                                     sl->mb_luma_dc[p],

                                                     h->dequant4_coeff[p][qscale][0]);

            else {

                static const uint8_t dc_mapping[16] = {

                     0 * 16,  1 * 16,  4 * 16,  5 * 16,

                     2 * 16,  3 * 16,  6 * 16,  7 * 16,

                     8 * 16,  9 * 16, 12 * 16, 13 * 16,

                    10 * 16, 11 * 16, 14 * 16, 15 * 16

                };

                for (i = 0; i < 16; i++)

                    dctcoef_set(sl->mb + (p * 256 << pixel_shift),

                                pixel_shift, dc_mapping[i],

                                dctcoef_get(sl->mb_luma_dc[p],

                                            pixel_shift, i));

            }

        }

    }

}
