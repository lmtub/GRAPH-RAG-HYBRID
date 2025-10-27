static void mpeg4_encode_vol_header(MpegEncContext * s, int vo_number, int vol_number)

{

    int vo_ver_id;



    if(s->max_b_frames || s->quarter_sample){

        vo_ver_id= 5;

        s->vo_type= ADV_SIMPLE_VO_TYPE;

    }else{

        vo_ver_id= 1;

        s->vo_type= SIMPLE_VO_TYPE;

    }



    put_bits(&s->pb, 16, 0);

    put_bits(&s->pb, 16, 0x100 + vo_number);        /* video obj */

    put_bits(&s->pb, 16, 0);

    put_bits(&s->pb, 16, 0x120 + vol_number);       /* video obj layer */



    put_bits(&s->pb, 1, 0);		/* random access vol */

    put_bits(&s->pb, 8, s->vo_type);	/* video obj type indication */

    put_bits(&s->pb, 1, 1);		/* is obj layer id= yes */

      put_bits(&s->pb, 4, vo_ver_id);	/* is obj layer ver id */

      put_bits(&s->pb, 3, 1);		/* is obj layer priority */

    

    aspect_to_info(s, s->avctx->sample_aspect_ratio);



    put_bits(&s->pb, 4, s->aspect_ratio_info);/* aspect ratio info */

    if (s->aspect_ratio_info == FF_ASPECT_EXTENDED){

        put_bits(&s->pb, 8, s->avctx->sample_aspect_ratio.num);

        put_bits(&s->pb, 8, s->avctx->sample_aspect_ratio.den);

    }



    if(s->low_delay){

        put_bits(&s->pb, 1, 1);		/* vol control parameters= yes */

        put_bits(&s->pb, 2, 1);		/* chroma format YUV 420/YV12 */

        put_bits(&s->pb, 1, s->low_delay);

        put_bits(&s->pb, 1, 0);		/* vbv parameters= no */

    }else{

        put_bits(&s->pb, 1, 0);		/* vol control parameters= no */

    }



    put_bits(&s->pb, 2, RECT_SHAPE);	/* vol shape= rectangle */

    put_bits(&s->pb, 1, 1);		/* marker bit */

    

    put_bits(&s->pb, 16, s->time_increment_resolution);

    if (s->time_increment_bits < 1)

        s->time_increment_bits = 1;

    put_bits(&s->pb, 1, 1);		/* marker bit */

    put_bits(&s->pb, 1, 0);		/* fixed vop rate=no */

    put_bits(&s->pb, 1, 1);		/* marker bit */

    put_bits(&s->pb, 13, s->width);	/* vol width */

    put_bits(&s->pb, 1, 1);		/* marker bit */

    put_bits(&s->pb, 13, s->height);	/* vol height */

    put_bits(&s->pb, 1, 1);		/* marker bit */

    put_bits(&s->pb, 1, s->progressive_sequence ? 0 : 1);

    put_bits(&s->pb, 1, 1);		/* obmc disable */

    if (vo_ver_id == 1) {

        put_bits(&s->pb, 1, s->vol_sprite_usage=0);		/* sprite enable */

    }else{

        put_bits(&s->pb, 2, s->vol_sprite_usage=0);		/* sprite enable */

    }

    

    s->quant_precision=5;

    put_bits(&s->pb, 1, 0);		/* not 8 bit == false */

    put_bits(&s->pb, 1, s->mpeg_quant);	/* quant type= (0=h263 style)*/



    if(s->mpeg_quant){

        ff_write_quant_matrix(&s->pb, s->avctx->intra_matrix);

        ff_write_quant_matrix(&s->pb, s->avctx->inter_matrix);

    }



    if (vo_ver_id != 1)

        put_bits(&s->pb, 1, s->quarter_sample);

    put_bits(&s->pb, 1, 1);		/* complexity estimation disable */

    s->resync_marker= s->rtp_mode;

    put_bits(&s->pb, 1, s->resync_marker ? 0 : 1);/* resync marker disable */

    put_bits(&s->pb, 1, s->data_partitioning ? 1 : 0);

    if(s->data_partitioning){

        put_bits(&s->pb, 1, 0);		/* no rvlc */

    }



    if (vo_ver_id != 1){

        put_bits(&s->pb, 1, 0);		/* newpred */

        put_bits(&s->pb, 1, 0);		/* reduced res vop */

    }

    put_bits(&s->pb, 1, 0);		/* scalability */

    

    ff_mpeg4_stuffing(&s->pb);



    /* user data */

    if(!(s->flags & CODEC_FLAG_BITEXACT)){

        put_bits(&s->pb, 16, 0);

        put_bits(&s->pb, 16, 0x1B2);	/* user_data */

	put_string(&s->pb, LIBAVCODEC_IDENT);

        ff_mpeg4_stuffing(&s->pb);

    }

}
