decode_cabac_residual_internal(const H264Context *h, H264SliceContext *sl,

                               int16_t *block,

                               int cat, int n, const uint8_t *scantable,

                               const uint32_t *qmul, int max_coeff,

                               int is_dc, int chroma422)

{

    static const int significant_coeff_flag_offset[2][14] = {

      { 105+0, 105+15, 105+29, 105+44, 105+47, 402, 484+0, 484+15, 484+29, 660, 528+0, 528+15, 528+29, 718 },

      { 277+0, 277+15, 277+29, 277+44, 277+47, 436, 776+0, 776+15, 776+29, 675, 820+0, 820+15, 820+29, 733 }

    };

    static const int last_coeff_flag_offset[2][14] = {

      { 166+0, 166+15, 166+29, 166+44, 166+47, 417, 572+0, 572+15, 572+29, 690, 616+0, 616+15, 616+29, 748 },

      { 338+0, 338+15, 338+29, 338+44, 338+47, 451, 864+0, 864+15, 864+29, 699, 908+0, 908+15, 908+29, 757 }

    };

    static const int coeff_abs_level_m1_offset[14] = {

        227+0, 227+10, 227+20, 227+30, 227+39, 426, 952+0, 952+10, 952+20, 708, 982+0, 982+10, 982+20, 766

    };

    static const uint8_t significant_coeff_flag_offset_8x8[2][63] = {

      { 0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5,

        4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9,10, 9, 8, 7,

        7, 6,11,12,13,11, 6, 7, 8, 9,14,10, 9, 8, 6,11,

       12,13,11, 6, 9,14,10, 9,11,12,13,11,14,10,12 },

      { 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 4, 5,

        6, 9,10,10, 8,11,12,11, 9, 9,10,10, 8,11,12,11,

        9, 9,10,10, 8,11,12,11, 9, 9,10,10, 8,13,13, 9,

        9,10,10, 8,13,13, 9, 9,10,10,14,14,14,14,14 }

    };

    static const uint8_t sig_coeff_offset_dc[7] = { 0, 0, 1, 1, 2, 2, 2 };

    /* node ctx: 0..3: abslevel1 (with abslevelgt1 == 0).

     * 4..7: abslevelgt1 + 3 (and abslevel1 doesn't matter).

     * map node ctx => cabac ctx for level=1 */

    static const uint8_t coeff_abs_level1_ctx[8] = { 1, 2, 3, 4, 0, 0, 0, 0 };

    /* map node ctx => cabac ctx for level>1 */

    static const uint8_t coeff_abs_levelgt1_ctx[2][8] = {

        { 5, 5, 5, 5, 6, 7, 8, 9 },

        { 5, 5, 5, 5, 6, 7, 8, 8 }, // 422/dc case

    };

    static const uint8_t coeff_abs_level_transition[2][8] = {

    /* update node ctx after decoding a level=1 */

        { 1, 2, 3, 3, 4, 5, 6, 7 },

    /* update node ctx after decoding a level>1 */

        { 4, 4, 4, 4, 5, 6, 7, 7 }

    };



    int index[64];



    int last;

    int coeff_count = 0;

    int node_ctx = 0;



    uint8_t *significant_coeff_ctx_base;

    uint8_t *last_coeff_ctx_base;

    uint8_t *abs_level_m1_ctx_base;



#if !ARCH_X86

#define CABAC_ON_STACK

#endif

#ifdef CABAC_ON_STACK

#define CC &cc

    CABACContext cc;

    cc.range     = sl->cabac.range;

    cc.low       = sl->cabac.low;

    cc.bytestream= sl->cabac.bytestream;

#if !UNCHECKED_BITSTREAM_READER || ARCH_AARCH64

    cc.bytestream_end = sl->cabac.bytestream_end;

#endif

#else

#define CC &sl->cabac

#endif



    significant_coeff_ctx_base = sl->cabac_state

        + significant_coeff_flag_offset[MB_FIELD(sl)][cat];

    last_coeff_ctx_base = sl->cabac_state

        + last_coeff_flag_offset[MB_FIELD(sl)][cat];

    abs_level_m1_ctx_base = sl->cabac_state

        + coeff_abs_level_m1_offset[cat];



    if( !is_dc && max_coeff == 64 ) {

#define DECODE_SIGNIFICANCE( coefs, sig_off, last_off ) \

        for(last= 0; last < coefs; last++) { \

            uint8_t *sig_ctx = significant_coeff_ctx_base + sig_off; \

            if( get_cabac( CC, sig_ctx )) { \

                uint8_t *last_ctx = last_coeff_ctx_base + last_off; \

                index[coeff_count++] = last; \

                if( get_cabac( CC, last_ctx ) ) { \

                    last= max_coeff; \

                    break; \

                } \

            } \

        }\

        if( last == max_coeff -1 ) {\

            index[coeff_count++] = last;\

        }

        const uint8_t *sig_off = significant_coeff_flag_offset_8x8[MB_FIELD(sl)];

#ifdef decode_significance

        coeff_count = decode_significance_8x8(CC, significant_coeff_ctx_base, index,

                                                 last_coeff_ctx_base, sig_off);

    } else {

        if (is_dc && chroma422) { // dc 422

            DECODE_SIGNIFICANCE(7, sig_coeff_offset_dc[last], sig_coeff_offset_dc[last]);

        } else {

            coeff_count = decode_significance(CC, max_coeff, significant_coeff_ctx_base, index,

                                                 last_coeff_ctx_base-significant_coeff_ctx_base);

        }

#else

        DECODE_SIGNIFICANCE( 63, sig_off[last], ff_h264_last_coeff_flag_offset_8x8[last] );

    } else {

        if (is_dc && chroma422) { // dc 422

            DECODE_SIGNIFICANCE(7, sig_coeff_offset_dc[last], sig_coeff_offset_dc[last]);

        } else {

            DECODE_SIGNIFICANCE(max_coeff - 1, last, last);

        }

#endif

    }

    av_assert2(coeff_count > 0);



    if( is_dc ) {

        if( cat == 3 )

            h->cbp_table[sl->mb_xy] |= 0x40 << (n - CHROMA_DC_BLOCK_INDEX);

        else

            h->cbp_table[sl->mb_xy] |= 0x100 << (n - LUMA_DC_BLOCK_INDEX);

        sl->non_zero_count_cache[scan8[n]] = coeff_count;

    } else {

        if( max_coeff == 64 )

            fill_rectangle(&sl->non_zero_count_cache[scan8[n]], 2, 2, 8, coeff_count, 1);

        else {

            av_assert2( cat == 1 || cat ==  2 || cat ==  4 || cat == 7 || cat == 8 || cat == 11 || cat == 12 );

            sl->non_zero_count_cache[scan8[n]] = coeff_count;

        }

    }



#define STORE_BLOCK(type) \

    do { \

        uint8_t *ctx = coeff_abs_level1_ctx[node_ctx] + abs_level_m1_ctx_base; \

 \

        int j= scantable[index[--coeff_count]]; \

 \

        if( get_cabac( CC, ctx ) == 0 ) { \

            node_ctx = coeff_abs_level_transition[0][node_ctx]; \

            if( is_dc ) { \

                ((type*)block)[j] = get_cabac_bypass_sign( CC, -1); \

            }else{ \

                ((type*)block)[j] = (get_cabac_bypass_sign( CC, -qmul[j]) + 32) >> 6; \

            } \

        } else { \

            int coeff_abs = 2; \

            ctx = coeff_abs_levelgt1_ctx[is_dc && chroma422][node_ctx] + abs_level_m1_ctx_base; \

            node_ctx = coeff_abs_level_transition[1][node_ctx]; \

\

            while( coeff_abs < 15 && get_cabac( CC, ctx ) ) { \

                coeff_abs++; \

            } \

\

            if( coeff_abs >= 15 ) { \

                int j = 0; \

                while (get_cabac_bypass(CC) && j < 30) { \

                    j++; \

                } \

\

                coeff_abs=1; \

                while( j-- ) { \

                    coeff_abs += coeff_abs + get_cabac_bypass( CC ); \

                } \

                coeff_abs+= 14U; \

            } \

\

            if( is_dc ) { \

                ((type*)block)[j] = get_cabac_bypass_sign( CC, -coeff_abs ); \

            }else{ \

                ((type*)block)[j] = ((int)(get_cabac_bypass_sign( CC, -coeff_abs ) * qmul[j] + 32)) >> 6; \

            } \

        } \

    } while ( coeff_count );



    if (h->pixel_shift) {

        STORE_BLOCK(int32_t)

    } else {

        STORE_BLOCK(int16_t)

    }

#ifdef CABAC_ON_STACK

            sl->cabac.range     = cc.range     ;

            sl->cabac.low       = cc.low       ;

            sl->cabac.bytestream= cc.bytestream;

#endif



}
