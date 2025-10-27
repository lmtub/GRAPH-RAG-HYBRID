int decode_luma_residual(const H264Context *h, H264SliceContext *sl,

                         GetBitContext *gb, const uint8_t *scan,

                         const uint8_t *scan8x8, int pixel_shift,

                         int mb_type, int cbp, int p)

{

    int i4x4, i8x8;

    int qscale = p == 0 ? sl->qscale : sl->chroma_qp[p - 1];

    if(IS_INTRA16x16(mb_type)){

        AV_ZERO128(sl->mb_luma_dc[p]+0);

        AV_ZERO128(sl->mb_luma_dc[p]+8);

        AV_ZERO128(sl->mb_luma_dc[p]+16);

        AV_ZERO128(sl->mb_luma_dc[p]+24);

        if (decode_residual(h, sl, gb, sl->mb_luma_dc[p], LUMA_DC_BLOCK_INDEX + p, scan, NULL, 16) < 0) {

            return -1; //FIXME continue if partitioned and other return -1 too

        }



        assert((cbp&15) == 0 || (cbp&15) == 15);



        if(cbp&15){

            for(i8x8=0; i8x8<4; i8x8++){

                for(i4x4=0; i4x4<4; i4x4++){

                    const int index= i4x4 + 4*i8x8 + p*16;

                    if( decode_residual(h, sl, gb, sl->mb + (16*index << pixel_shift),

                        index, scan + 1, h->dequant4_coeff[p][qscale], 15) < 0 ){

                        return -1;

                    }

                }

            }

            return 0xf;

        }else{

            fill_rectangle(&sl->non_zero_count_cache[scan8[p*16]], 4, 4, 8, 0, 1);

            return 0;

        }

    }else{

        int cqm = (IS_INTRA( mb_type ) ? 0:3)+p;

        /* For CAVLC 4:4:4, we need to keep track of the luma 8x8 CBP for deblocking nnz purposes. */

        int new_cbp = 0;

        for(i8x8=0; i8x8<4; i8x8++){

            if(cbp & (1<<i8x8)){

                if(IS_8x8DCT(mb_type)){

                    int16_t *buf = &sl->mb[64*i8x8+256*p << pixel_shift];

                    uint8_t *nnz;

                    for(i4x4=0; i4x4<4; i4x4++){

                        const int index= i4x4 + 4*i8x8 + p*16;

                        if( decode_residual(h, sl, gb, buf, index, scan8x8+16*i4x4,

                                            h->dequant8_coeff[cqm][qscale], 16) < 0 )

                            return -1;

                    }

                    nnz = &sl->non_zero_count_cache[scan8[4 * i8x8 + p * 16]];

                    nnz[0] += nnz[1] + nnz[8] + nnz[9];

                    new_cbp |= !!nnz[0] << i8x8;

                }else{

                    for(i4x4=0; i4x4<4; i4x4++){

                        const int index= i4x4 + 4*i8x8 + p*16;

                        if( decode_residual(h, sl, gb, sl->mb + (16*index << pixel_shift), index,

                                            scan, h->dequant4_coeff[cqm][qscale], 16) < 0 ){

                            return -1;

                        }

                        new_cbp |= sl->non_zero_count_cache[scan8[index]] << i8x8;

                    }

                }

            }else{

                uint8_t * const nnz = &sl->non_zero_count_cache[scan8[4 * i8x8 + p * 16]];

                nnz[0] = nnz[1] = nnz[8] = nnz[9] = 0;

            }

        }

        return new_cbp;

    }

}
