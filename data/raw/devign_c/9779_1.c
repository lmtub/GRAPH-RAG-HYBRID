int ff_msmpeg4_decode_init(MpegEncContext *s)

{

    static int done = 0;

    int i;

    MVTable *mv;



    common_init(s);



    if (!done) {

        done = 1;



        for(i=0;i<NB_RL_TABLES;i++) {

            init_rl(&rl_table[i]);

            init_vlc_rl(&rl_table[i]);

        }

        for(i=0;i<2;i++) {

            mv = &mv_tables[i];

            init_vlc(&mv->vlc, MV_VLC_BITS, mv->n + 1, 

                     mv->table_mv_bits, 1, 1,

                     mv->table_mv_code, 2, 2);

        }



        init_vlc(&dc_lum_vlc[0], DC_VLC_BITS, 120, 

                 &table0_dc_lum[0][1], 8, 4,

                 &table0_dc_lum[0][0], 8, 4);

        init_vlc(&dc_chroma_vlc[0], DC_VLC_BITS, 120, 

                 &table0_dc_chroma[0][1], 8, 4,

                 &table0_dc_chroma[0][0], 8, 4);

        init_vlc(&dc_lum_vlc[1], DC_VLC_BITS, 120, 

                 &table1_dc_lum[0][1], 8, 4,

                 &table1_dc_lum[0][0], 8, 4);

        init_vlc(&dc_chroma_vlc[1], DC_VLC_BITS, 120, 

                 &table1_dc_chroma[0][1], 8, 4,

                 &table1_dc_chroma[0][0], 8, 4);

    

        init_vlc(&v2_dc_lum_vlc, DC_VLC_BITS, 512, 

                 &v2_dc_lum_table[0][1], 8, 4,

                 &v2_dc_lum_table[0][0], 8, 4);

        init_vlc(&v2_dc_chroma_vlc, DC_VLC_BITS, 512, 

                 &v2_dc_chroma_table[0][1], 8, 4,

                 &v2_dc_chroma_table[0][0], 8, 4);

    

        init_vlc(&cbpy_vlc, CBPY_VLC_BITS, 16,

                 &cbpy_tab[0][1], 2, 1,

                 &cbpy_tab[0][0], 2, 1);

        init_vlc(&v2_intra_cbpc_vlc, V2_INTRA_CBPC_VLC_BITS, 4,

                 &v2_intra_cbpc[0][1], 2, 1,

                 &v2_intra_cbpc[0][0], 2, 1);

        init_vlc(&v2_mb_type_vlc, V2_MB_TYPE_VLC_BITS, 8,

                 &v2_mb_type[0][1], 2, 1,

                 &v2_mb_type[0][0], 2, 1);

        init_vlc(&v2_mv_vlc, V2_MV_VLC_BITS, 33,

                 &mvtab[0][1], 2, 1,

                 &mvtab[0][0], 2, 1);



        for(i=0; i<4; i++){

            init_vlc(&mb_non_intra_vlc[i], MB_NON_INTRA_VLC_BITS, 128, 

                     &wmv2_inter_table[i][0][1], 8, 4,

                     &wmv2_inter_table[i][0][0], 8, 4); //FIXME name?

        }

        

        init_vlc(&mb_intra_vlc, MB_INTRA_VLC_BITS, 64, 

                 &table_mb_intra[0][1], 4, 2,

                 &table_mb_intra[0][0], 4, 2);

        

        init_vlc(&v1_intra_cbpc_vlc, V1_INTRA_CBPC_VLC_BITS, 8, 

                 intra_MCBPC_bits, 1, 1,

                 intra_MCBPC_code, 1, 1);

        init_vlc(&v1_inter_cbpc_vlc, V1_INTER_CBPC_VLC_BITS, 25, 

                 inter_MCBPC_bits, 1, 1,

                 inter_MCBPC_code, 1, 1);

        

        init_vlc(&inter_intra_vlc, INTER_INTRA_VLC_BITS, 4, 

                 &table_inter_intra[0][1], 2, 1,

                 &table_inter_intra[0][0], 2, 1);

    }

    

    switch(s->msmpeg4_version){

    case 1:

    case 2:

        s->decode_mb= msmpeg4v12_decode_mb;

        break;

    case 3:

    case 4:

        s->decode_mb= msmpeg4v34_decode_mb;

        break;

    case 5:

        s->decode_mb= wmv2_decode_mb;

        break;

    }

    

    s->slice_height= s->mb_height; //to avoid 1/0 if the first frame isnt a keyframe

    

    return 0;

}
