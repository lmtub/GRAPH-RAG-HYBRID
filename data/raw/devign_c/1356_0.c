static av_always_inline void decode_cabac_luma_residual(const H264Context *h, H264SliceContext *sl,

                                                        const uint8_t *scan, const uint8_t *scan8x8,

                                                        int pixel_shift, int mb_type, int cbp, int p)

{

    static const uint8_t ctx_cat[4][3] = {{0,6,10},{1,7,11},{2,8,12},{5,9,13}};

    const uint32_t *qmul;

    int i8x8, i4x4;

    int qscale = p == 0 ? sl->qscale : sl->chroma_qp[p - 1];

    if( IS_INTRA16x16( mb_type ) ) {

        AV_ZERO128(sl->mb_luma_dc[p]+0);

        AV_ZERO128(sl->mb_luma_dc[p]+8);

        AV_ZERO128(sl->mb_luma_dc[p]+16);

        AV_ZERO128(sl->mb_luma_dc[p]+24);

        decode_cabac_residual_dc(h, sl, sl->mb_luma_dc[p], ctx_cat[0][p], LUMA_DC_BLOCK_INDEX+p, scan, 16);



        if( cbp&15 ) {

            qmul = h->dequant4_coeff[p][qscale];

            for( i4x4 = 0; i4x4 < 16; i4x4++ ) {

                const int index = 16*p + i4x4;

                decode_cabac_residual_nondc(h, sl, sl->mb + (16*index << pixel_shift), ctx_cat[1][p], index, scan + 1, qmul, 15);

            }

        } else {

            fill_rectangle(&sl->non_zero_count_cache[scan8[16*p]], 4, 4, 8, 0, 1);

        }

    } else {

        int cqm = (IS_INTRA( mb_type ) ? 0:3) + p;

        for( i8x8 = 0; i8x8 < 4; i8x8++ ) {

            if( cbp & (1<<i8x8) ) {

                if( IS_8x8DCT(mb_type) ) {

                    const int index = 16*p + 4*i8x8;

                    decode_cabac_residual_nondc(h, sl, sl->mb + (16*index << pixel_shift), ctx_cat[3][p], index,

                                                scan8x8, h->dequant8_coeff[cqm][qscale], 64);

                } else {

                    qmul = h->dequant4_coeff[cqm][qscale];

                    for( i4x4 = 0; i4x4 < 4; i4x4++ ) {

                        const int index = 16*p + 4*i8x8 + i4x4;

//START_TIMER

                        decode_cabac_residual_nondc(h, sl, sl->mb + (16*index << pixel_shift), ctx_cat[2][p], index, scan, qmul, 16);

//STOP_TIMER("decode_residual")

                    }

                }

            } else {

                fill_rectangle(&sl->non_zero_count_cache[scan8[4*i8x8+16*p]], 2, 2, 8, 0, 1);

            }

        }

    }

}
