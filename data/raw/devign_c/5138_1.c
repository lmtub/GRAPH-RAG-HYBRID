static int vc1_init_common(VC1Context *v)

{

    static int done = 0;

    int i = 0;



    v->hrd_rate = v->hrd_buffer = NULL;



    /* VLC tables */

    if(!done)

    {

        done = 1;

        init_vlc(&ff_vc1_bfraction_vlc, VC1_BFRACTION_VLC_BITS, 23,

                 ff_vc1_bfraction_bits, 1, 1,

                 ff_vc1_bfraction_codes, 1, 1, INIT_VLC_USE_STATIC);

        init_vlc(&ff_vc1_norm2_vlc, VC1_NORM2_VLC_BITS, 4,

                 ff_vc1_norm2_bits, 1, 1,

                 ff_vc1_norm2_codes, 1, 1, INIT_VLC_USE_STATIC);

        init_vlc(&ff_vc1_norm6_vlc, VC1_NORM6_VLC_BITS, 64,

                 ff_vc1_norm6_bits, 1, 1,

                 ff_vc1_norm6_codes, 2, 2, INIT_VLC_USE_STATIC);

        init_vlc(&ff_vc1_imode_vlc, VC1_IMODE_VLC_BITS, 7,

                 ff_vc1_imode_bits, 1, 1,

                 ff_vc1_imode_codes, 1, 1, INIT_VLC_USE_STATIC);

        for (i=0; i<3; i++)

        {

            init_vlc(&ff_vc1_ttmb_vlc[i], VC1_TTMB_VLC_BITS, 16,

                     ff_vc1_ttmb_bits[i], 1, 1,

                     ff_vc1_ttmb_codes[i], 2, 2, INIT_VLC_USE_STATIC);

            init_vlc(&ff_vc1_ttblk_vlc[i], VC1_TTBLK_VLC_BITS, 8,

                     ff_vc1_ttblk_bits[i], 1, 1,

                     ff_vc1_ttblk_codes[i], 1, 1, INIT_VLC_USE_STATIC);

            init_vlc(&ff_vc1_subblkpat_vlc[i], VC1_SUBBLKPAT_VLC_BITS, 15,

                     ff_vc1_subblkpat_bits[i], 1, 1,

                     ff_vc1_subblkpat_codes[i], 1, 1, INIT_VLC_USE_STATIC);

        }

        for(i=0; i<4; i++)

        {

            init_vlc(&ff_vc1_4mv_block_pattern_vlc[i], VC1_4MV_BLOCK_PATTERN_VLC_BITS, 16,

                     ff_vc1_4mv_block_pattern_bits[i], 1, 1,

                     ff_vc1_4mv_block_pattern_codes[i], 1, 1, INIT_VLC_USE_STATIC);

            init_vlc(&ff_vc1_cbpcy_p_vlc[i], VC1_CBPCY_P_VLC_BITS, 64,

                     ff_vc1_cbpcy_p_bits[i], 1, 1,

                     ff_vc1_cbpcy_p_codes[i], 2, 2, INIT_VLC_USE_STATIC);

            init_vlc(&ff_vc1_mv_diff_vlc[i], VC1_MV_DIFF_VLC_BITS, 73,

                     ff_vc1_mv_diff_bits[i], 1, 1,

                     ff_vc1_mv_diff_codes[i], 2, 2, INIT_VLC_USE_STATIC);

        }

        for(i=0; i<8; i++)

            init_vlc(&ff_vc1_ac_coeff_table[i], AC_VLC_BITS, vc1_ac_sizes[i],

                     &vc1_ac_tables[i][0][1], 8, 4,

                     &vc1_ac_tables[i][0][0], 8, 4, INIT_VLC_USE_STATIC);

        init_vlc(&ff_msmp4_mb_i_vlc, MB_INTRA_VLC_BITS, 64,

                 &ff_msmp4_mb_i_table[0][1], 4, 2,

                 &ff_msmp4_mb_i_table[0][0], 4, 2, INIT_VLC_USE_STATIC);

    }



    /* Other defaults */

    v->pq = -1;

    v->mvrange = 0; /* 7.1.1.18, p80 */



    return 0;

}
