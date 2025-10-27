static int vc9_init_common(VC9Context *v)

{

    static int done = 0;

    int i;



    /* Set the bit planes */

    v->mv_type_mb_plane = (struct BitPlane) { NULL, 0, 0, 0 };

    v->direct_mb_plane = (struct BitPlane) { NULL, 0, 0, 0 };

    v->skip_mb_plane = (struct BitPlane) { NULL, 0, 0, 0 };

#if HAS_ADVANCED_PROFILE

    v->ac_pred_plane = v->over_flags_plane = (struct BitPlane) { NULL, 0, 0, 0 };

    v->hrd_rate = v->hrd_buffer = NULL;

#endif



    /* VLC tables */

#if VLC_NORM6_METH0D == 1

#  if 0 // spec -> actual tables converter

    for(i=0; i<64; i++){

        int code= (vc9_norm6_spec[i][1] << vc9_norm6_spec[i][4]) + vc9_norm6_spec[i][3];

        av_log(NULL, AV_LOG_DEBUG, "0x%03X, ", code);

        if(i%16==15) av_log(NULL, AV_LOG_DEBUG, "\n");

    }

    for(i=0; i<64; i++){

        int code= vc9_norm6_spec[i][2] + vc9_norm6_spec[i][4];

        av_log(NULL, AV_LOG_DEBUG, "%2d, ", code);

        if(i%16==15) av_log(NULL, AV_LOG_DEBUG, "\n");

    }

#  endif

#endif

    if(!done)

    {

        done = 1;

        INIT_VLC(&vc9_bfraction_vlc, VC9_BFRACTION_VLC_BITS, 23,

                 vc9_bfraction_bits, 1, 1,

                 vc9_bfraction_codes, 1, 1, 1);

        INIT_VLC(&vc9_norm2_vlc, VC9_NORM2_VLC_BITS, 4,

                 vc9_norm2_bits, 1, 1,

                 vc9_norm2_codes, 1, 1, 1);

#if VLC_NORM6_METH0D == 1

        INIT_VLC(&vc9_norm6_vlc, VC9_NORM6_VLC_BITS, 64,

                 vc9_norm6_bits, 1, 1,

                 vc9_norm6_codes, 2, 2, 1);

#endif

#if VLC_NORM6_METH0D == 2

        INIT_VLC(&vc9_norm6_first_vlc, VC9_NORM6_FIRST_BITS, 24,

                 &vc9_norm6_first[0][1], 1, 1,

                 &vc9_norm6_first[0][0], 1, 1, 1);

        INIT_VLC(&vc9_norm6_second_vlc, VC9_NORM6_SECOND_BITS, 22,

                 &vc9_norm6_second[0][1], 1, 1,

                 &vc9_norm6_second[0][0], 1, 1, 1);

#endif

        INIT_VLC(&vc9_imode_vlc, VC9_IMODE_VLC_BITS, 7,

                 vc9_imode_bits, 1, 1,

                 vc9_imode_codes, 1, 1, 1);

        for (i=0; i<3; i++)

        {

            INIT_VLC(&vc9_ttmb_vlc[i], VC9_TTMB_VLC_BITS, 16,

                     vc9_ttmb_bits[i], 1, 1,

                     vc9_ttmb_codes[i], 2, 2, 1);

        }

        for(i=0; i<4; i++)

        {

            INIT_VLC(&vc9_4mv_block_pattern_vlc[i], VC9_4MV_BLOCK_PATTERN_VLC_BITS, 16,

                     vc9_4mv_block_pattern_bits[i], 1, 1,

                     vc9_4mv_block_pattern_codes[i], 1, 1, 1);

            INIT_VLC(&vc9_cbpcy_p_vlc[i], VC9_CBPCY_P_VLC_BITS, 64,

                     vc9_cbpcy_p_bits[i], 1, 1,

                     vc9_cbpcy_p_codes[i], 2, 2, 1);

            INIT_VLC(&vc9_mv_diff_vlc[i], VC9_MV_DIFF_VLC_BITS, 73,

                     vc9_mv_diff_bits[i], 1, 1,

                     vc9_mv_diff_codes[i], 2, 2, 1);

        }

    }



    /* Other defaults */

    v->pq = -1;

    v->mvrange = 0; /* 7.1.1.18, p80 */



    return 0;

}
