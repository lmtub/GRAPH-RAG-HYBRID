void h263_encode_init(MpegEncContext *s)

{

    static int done = 0;



    if (!done) {

        done = 1;



        init_uni_dc_tab();



        init_rl(&rl_inter);

        init_rl(&rl_intra);

        init_rl(&rl_intra_aic);

        

        init_uni_mpeg4_rl_tab(&rl_intra, uni_mpeg4_intra_rl_bits, uni_mpeg4_intra_rl_len);

        init_uni_mpeg4_rl_tab(&rl_inter, uni_mpeg4_inter_rl_bits, uni_mpeg4_inter_rl_len);



        init_uni_h263_rl_tab(&rl_intra_aic, NULL, uni_h263_intra_aic_rl_len);

        init_uni_h263_rl_tab(&rl_inter    , NULL, uni_h263_inter_rl_len);



        init_mv_penalty_and_fcode(s);

    }

    s->me.mv_penalty= mv_penalty; //FIXME exact table for msmpeg4 & h263p

    

    s->intra_ac_vlc_length     =s->inter_ac_vlc_length     = uni_h263_inter_rl_len;

    s->intra_ac_vlc_last_length=s->inter_ac_vlc_last_length= uni_h263_inter_rl_len + 128*64;

    if(s->h263_aic){

        s->intra_ac_vlc_length     = uni_h263_intra_aic_rl_len;

        s->intra_ac_vlc_last_length= uni_h263_intra_aic_rl_len + 128*64;

    }

    s->ac_esc_length= 7+1+6+8;



    // use fcodes >1 only for mpeg4 & h263 & h263p FIXME

    switch(s->codec_id){

    case CODEC_ID_MPEG4:

        s->fcode_tab= fcode_tab;

        s->min_qcoeff= -2048;

        s->max_qcoeff=  2047;

        s->intra_ac_vlc_length     = uni_mpeg4_intra_rl_len;

        s->intra_ac_vlc_last_length= uni_mpeg4_intra_rl_len + 128*64;

        s->inter_ac_vlc_length     = uni_mpeg4_inter_rl_len;

        s->inter_ac_vlc_last_length= uni_mpeg4_inter_rl_len + 128*64;

        s->luma_dc_vlc_length= uni_DCtab_lum_len;

        s->chroma_dc_vlc_length= uni_DCtab_chrom_len;

        s->ac_esc_length= 7+2+1+6+1+12+1;

        s->y_dc_scale_table= ff_mpeg4_y_dc_scale_table;

        s->c_dc_scale_table= ff_mpeg4_c_dc_scale_table;



        if(s->flags & CODEC_FLAG_GLOBAL_HEADER){



            s->avctx->extradata= av_malloc(1024);

            init_put_bits(&s->pb, s->avctx->extradata, 1024);

            

            mpeg4_encode_visual_object_header(s);

            mpeg4_encode_vol_header(s, 0, 0);



//            ff_mpeg4_stuffing(&s->pb); ?

            flush_put_bits(&s->pb);

            s->avctx->extradata_size= (put_bits_count(&s->pb)+7)>>3;

        }

        

        break;

    case CODEC_ID_H263P:

        if(s->umvplus)

            s->fcode_tab= umv_fcode_tab;

        if(s->modified_quant){

            s->min_qcoeff= -2047;

            s->max_qcoeff=  2047;

        }else{

            s->min_qcoeff= -127;

            s->max_qcoeff=  127;

        }

        break;

        //Note for mpeg4 & h263 the dc-scale table will be set per frame as needed later 

    case CODEC_ID_FLV1:

        if (s->h263_flv > 1) {

            s->min_qcoeff= -1023;

            s->max_qcoeff=  1023;

        } else {

            s->min_qcoeff= -127;

            s->max_qcoeff=  127;

        }

        s->y_dc_scale_table=

        s->c_dc_scale_table= ff_mpeg1_dc_scale_table;

        break;

    default: //nothing needed default table allready set in mpegvideo.c

        s->min_qcoeff= -127;

        s->max_qcoeff=  127;

        s->y_dc_scale_table=

        s->c_dc_scale_table= ff_mpeg1_dc_scale_table;

    }

}
