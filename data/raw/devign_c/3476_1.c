void h263_decode_init_vlc(MpegEncContext *s)

{

    static int done = 0;



    if (!done) {

        done = 1;



        init_vlc(&intra_MCBPC_vlc, INTRA_MCBPC_VLC_BITS, 9, 

                 intra_MCBPC_bits, 1, 1,

                 intra_MCBPC_code, 1, 1);

        init_vlc(&inter_MCBPC_vlc, INTER_MCBPC_VLC_BITS, 28, 

                 inter_MCBPC_bits, 1, 1,

                 inter_MCBPC_code, 1, 1);

        init_vlc(&cbpy_vlc, CBPY_VLC_BITS, 16,

                 &cbpy_tab[0][1], 2, 1,

                 &cbpy_tab[0][0], 2, 1);

        init_vlc(&mv_vlc, MV_VLC_BITS, 33,

                 &mvtab[0][1], 2, 1,

                 &mvtab[0][0], 2, 1);

        init_rl(&rl_inter);

        init_rl(&rl_intra);

        init_rl(&rvlc_rl_inter);

        init_rl(&rvlc_rl_intra);

        init_rl(&rl_intra_aic);

        init_vlc_rl(&rl_inter);

        init_vlc_rl(&rl_intra);

        init_vlc_rl(&rvlc_rl_inter);

        init_vlc_rl(&rvlc_rl_intra);

        init_vlc_rl(&rl_intra_aic);

        init_vlc(&dc_lum, DC_VLC_BITS, 10 /* 13 */,

                 &DCtab_lum[0][1], 2, 1,

                 &DCtab_lum[0][0], 2, 1);

        init_vlc(&dc_chrom, DC_VLC_BITS, 10 /* 13 */,

                 &DCtab_chrom[0][1], 2, 1,

                 &DCtab_chrom[0][0], 2, 1);

        init_vlc(&sprite_trajectory, SPRITE_TRAJ_VLC_BITS, 15,

                 &sprite_trajectory_tab[0][1], 4, 2,

                 &sprite_trajectory_tab[0][0], 4, 2);

        init_vlc(&mb_type_b_vlc, MB_TYPE_B_VLC_BITS, 4,

                 &mb_type_b_tab[0][1], 2, 1,

                 &mb_type_b_tab[0][0], 2, 1);

        init_vlc(&h263_mbtype_b_vlc, H263_MBTYPE_B_VLC_BITS, 15,

                 &h263_mbtype_b_tab[0][1], 2, 1,

                 &h263_mbtype_b_tab[0][0], 2, 1);

        init_vlc(&cbpc_b_vlc, CBPC_B_VLC_BITS, 4,

                 &cbpc_b_tab[0][1], 2, 1,

                 &cbpc_b_tab[0][0], 2, 1);

    }

}
