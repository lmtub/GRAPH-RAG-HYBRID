static int vc1_init_common(VC1Context *v)

{

    static int done = 0;

    int i = 0;

    static VLC_TYPE vlc_table[32372][2];



    v->hrd_rate = v->hrd_buffer = NULL;



    /* VLC tables */

    if (!done) {

        INIT_VLC_STATIC(&ff_vc1_bfraction_vlc, VC1_BFRACTION_VLC_BITS, 23,

                        ff_vc1_bfraction_bits, 1, 1,

                        ff_vc1_bfraction_codes, 1, 1, 1 << VC1_BFRACTION_VLC_BITS);

        INIT_VLC_STATIC(&ff_vc1_norm2_vlc, VC1_NORM2_VLC_BITS, 4,

                        ff_vc1_norm2_bits, 1, 1,

                        ff_vc1_norm2_codes, 1, 1, 1 << VC1_NORM2_VLC_BITS);

        INIT_VLC_STATIC(&ff_vc1_norm6_vlc, VC1_NORM6_VLC_BITS, 64,

                        ff_vc1_norm6_bits, 1, 1,

                        ff_vc1_norm6_codes, 2, 2, 556);

        INIT_VLC_STATIC(&ff_vc1_imode_vlc, VC1_IMODE_VLC_BITS, 7,

                        ff_vc1_imode_bits, 1, 1,

                        ff_vc1_imode_codes, 1, 1, 1 << VC1_IMODE_VLC_BITS);

        for (i = 0; i < 3; i++) {

            ff_vc1_ttmb_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 0]];

            ff_vc1_ttmb_vlc[i].table_allocated = vlc_offs[i * 3 + 1] - vlc_offs[i * 3 + 0];

            init_vlc(&ff_vc1_ttmb_vlc[i], VC1_TTMB_VLC_BITS, 16,

                     ff_vc1_ttmb_bits[i], 1, 1,

                     ff_vc1_ttmb_codes[i], 2, 2, INIT_VLC_USE_NEW_STATIC);

            ff_vc1_ttblk_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 1]];

            ff_vc1_ttblk_vlc[i].table_allocated = vlc_offs[i * 3 + 2] - vlc_offs[i * 3 + 1];

            init_vlc(&ff_vc1_ttblk_vlc[i], VC1_TTBLK_VLC_BITS, 8,

                     ff_vc1_ttblk_bits[i], 1, 1,

                     ff_vc1_ttblk_codes[i], 1, 1, INIT_VLC_USE_NEW_STATIC);

            ff_vc1_subblkpat_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 2]];

            ff_vc1_subblkpat_vlc[i].table_allocated = vlc_offs[i * 3 + 3] - vlc_offs[i * 3 + 2];

            init_vlc(&ff_vc1_subblkpat_vlc[i], VC1_SUBBLKPAT_VLC_BITS, 15,

                     ff_vc1_subblkpat_bits[i], 1, 1,

                     ff_vc1_subblkpat_codes[i], 1, 1, INIT_VLC_USE_NEW_STATIC);

        }

        for (i = 0; i < 4; i++) {

            ff_vc1_4mv_block_pattern_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 9]];

            ff_vc1_4mv_block_pattern_vlc[i].table_allocated = vlc_offs[i * 3 + 10] - vlc_offs[i * 3 + 9];

            init_vlc(&ff_vc1_4mv_block_pattern_vlc[i], VC1_4MV_BLOCK_PATTERN_VLC_BITS, 16,

                     ff_vc1_4mv_block_pattern_bits[i], 1, 1,

                     ff_vc1_4mv_block_pattern_codes[i], 1, 1, INIT_VLC_USE_NEW_STATIC);

            ff_vc1_cbpcy_p_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 10]];

            ff_vc1_cbpcy_p_vlc[i].table_allocated = vlc_offs[i * 3 + 11] - vlc_offs[i * 3 + 10];

            init_vlc(&ff_vc1_cbpcy_p_vlc[i], VC1_CBPCY_P_VLC_BITS, 64,

                     ff_vc1_cbpcy_p_bits[i], 1, 1,

                     ff_vc1_cbpcy_p_codes[i], 2, 2, INIT_VLC_USE_NEW_STATIC);

            ff_vc1_mv_diff_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 11]];

            ff_vc1_mv_diff_vlc[i].table_allocated = vlc_offs[i * 3 + 12] - vlc_offs[i * 3 + 11];

            init_vlc(&ff_vc1_mv_diff_vlc[i], VC1_MV_DIFF_VLC_BITS, 73,

                     ff_vc1_mv_diff_bits[i], 1, 1,

                     ff_vc1_mv_diff_codes[i], 2, 2, INIT_VLC_USE_NEW_STATIC);

        }

        for (i = 0; i < 8; i++) {

            ff_vc1_ac_coeff_table[i].table           = &vlc_table[vlc_offs[i * 2 + 21]];

            ff_vc1_ac_coeff_table[i].table_allocated = vlc_offs[i * 2 + 22] - vlc_offs[i * 2 + 21];

            init_vlc(&ff_vc1_ac_coeff_table[i], AC_VLC_BITS, vc1_ac_sizes[i],

                     &vc1_ac_tables[i][0][1], 8, 4,

                     &vc1_ac_tables[i][0][0], 8, 4, INIT_VLC_USE_NEW_STATIC);

            /* initialize interlaced MVDATA tables (2-Ref) */

            ff_vc1_2ref_mvdata_vlc[i].table           = &vlc_table[vlc_offs[i * 2 + 22]];

            ff_vc1_2ref_mvdata_vlc[i].table_allocated = vlc_offs[i * 2 + 23] - vlc_offs[i * 2 + 22];

            init_vlc(&ff_vc1_2ref_mvdata_vlc[i], VC1_2REF_MVDATA_VLC_BITS, 126,

                     ff_vc1_2ref_mvdata_bits[i], 1, 1,

                     ff_vc1_2ref_mvdata_codes[i], 4, 4, INIT_VLC_USE_NEW_STATIC);

        }

        for (i = 0; i < 4; i++) {

            /* initialize 4MV MBMODE VLC tables for interlaced frame P picture */

            ff_vc1_intfr_4mv_mbmode_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 37]];

            ff_vc1_intfr_4mv_mbmode_vlc[i].table_allocated = vlc_offs[i * 3 + 38] - vlc_offs[i * 3 + 37];

            init_vlc(&ff_vc1_intfr_4mv_mbmode_vlc[i], VC1_INTFR_4MV_MBMODE_VLC_BITS, 15,

                     ff_vc1_intfr_4mv_mbmode_bits[i], 1, 1,

                     ff_vc1_intfr_4mv_mbmode_codes[i], 2, 2, INIT_VLC_USE_NEW_STATIC);

            /* initialize NON-4MV MBMODE VLC tables for the same */

            ff_vc1_intfr_non4mv_mbmode_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 38]];

            ff_vc1_intfr_non4mv_mbmode_vlc[i].table_allocated = vlc_offs[i * 3 + 39] - vlc_offs[i * 3 + 38];

            init_vlc(&ff_vc1_intfr_non4mv_mbmode_vlc[i], VC1_INTFR_NON4MV_MBMODE_VLC_BITS, 9,

                     ff_vc1_intfr_non4mv_mbmode_bits[i], 1, 1,

                     ff_vc1_intfr_non4mv_mbmode_codes[i], 1, 1, INIT_VLC_USE_NEW_STATIC);

            /* initialize interlaced MVDATA tables (1-Ref) */

            ff_vc1_1ref_mvdata_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 39]];

            ff_vc1_1ref_mvdata_vlc[i].table_allocated = vlc_offs[i * 3 + 40] - vlc_offs[i * 3 + 39];

            init_vlc(&ff_vc1_1ref_mvdata_vlc[i], VC1_1REF_MVDATA_VLC_BITS, 72,

                     ff_vc1_1ref_mvdata_bits[i], 1, 1,

                     ff_vc1_1ref_mvdata_codes[i], 4, 4, INIT_VLC_USE_NEW_STATIC);

        }

        for (i = 0; i < 4; i++) {

            /* Initialize 2MV Block pattern VLC tables */

            ff_vc1_2mv_block_pattern_vlc[i].table           = &vlc_table[vlc_offs[i + 49]];

            ff_vc1_2mv_block_pattern_vlc[i].table_allocated = vlc_offs[i + 50] - vlc_offs[i + 49];

            init_vlc(&ff_vc1_2mv_block_pattern_vlc[i], VC1_2MV_BLOCK_PATTERN_VLC_BITS, 4,

                     ff_vc1_2mv_block_pattern_bits[i], 1, 1,

                     ff_vc1_2mv_block_pattern_codes[i], 1, 1, INIT_VLC_USE_NEW_STATIC);

        }

        for (i = 0; i < 8; i++) {

            /* Initialize interlaced CBPCY VLC tables (Table 124 - Table 131) */

            ff_vc1_icbpcy_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 53]];

            ff_vc1_icbpcy_vlc[i].table_allocated = vlc_offs[i * 3 + 54] - vlc_offs[i * 3 + 53];

            init_vlc(&ff_vc1_icbpcy_vlc[i], VC1_ICBPCY_VLC_BITS, 63,

                     ff_vc1_icbpcy_p_bits[i], 1, 1,

                     ff_vc1_icbpcy_p_codes[i], 2, 2, INIT_VLC_USE_NEW_STATIC);

            /* Initialize interlaced field picture MBMODE VLC tables */

            ff_vc1_if_mmv_mbmode_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 54]];

            ff_vc1_if_mmv_mbmode_vlc[i].table_allocated = vlc_offs[i * 3 + 55] - vlc_offs[i * 3 + 54];

            init_vlc(&ff_vc1_if_mmv_mbmode_vlc[i], VC1_IF_MMV_MBMODE_VLC_BITS, 8,

                     ff_vc1_if_mmv_mbmode_bits[i], 1, 1,

                     ff_vc1_if_mmv_mbmode_codes[i], 1, 1, INIT_VLC_USE_NEW_STATIC);

            ff_vc1_if_1mv_mbmode_vlc[i].table           = &vlc_table[vlc_offs[i * 3 + 55]];

            ff_vc1_if_1mv_mbmode_vlc[i].table_allocated = vlc_offs[i * 3 + 56] - vlc_offs[i * 3 + 55];

            init_vlc(&ff_vc1_if_1mv_mbmode_vlc[i], VC1_IF_1MV_MBMODE_VLC_BITS, 6,

                     ff_vc1_if_1mv_mbmode_bits[i], 1, 1,

                     ff_vc1_if_1mv_mbmode_codes[i], 1, 1, INIT_VLC_USE_NEW_STATIC);

        }

        done = 1;

    }



    /* Other defaults */

    v->pq      = -1;

    v->mvrange = 0; /* 7.1.1.18, p80 */



    return 0;

}
