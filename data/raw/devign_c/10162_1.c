static int rv20_decode_picture_header(RVDecContext *rv)

{

    MpegEncContext *s = &rv->m;

    int seq, mb_pos, i;

    int rpr_bits;



#if 0

    GetBitContext gb= s->gb;

    for(i=0; i<64; i++){

        av_log(s->avctx, AV_LOG_DEBUG, "%d", get_bits1(&gb));

        if(i%4==3) av_log(s->avctx, AV_LOG_DEBUG, " ");

    }

    av_log(s->avctx, AV_LOG_DEBUG, "\n");

#endif

#if 0

    av_log(s->avctx, AV_LOG_DEBUG, "%3dx%03d/%02Xx%02X ", s->width, s->height, s->width/4, s->height/4);

    for(i=0; i<s->avctx->extradata_size; i++){

        av_log(s->avctx, AV_LOG_DEBUG, "%02X ", ((uint8_t*)s->avctx->extradata)[i]);

        if(i%4==3) av_log(s->avctx, AV_LOG_DEBUG, " ");

    }

    av_log(s->avctx, AV_LOG_DEBUG, "\n");

#endif



    i= get_bits(&s->gb, 2);

    switch(i){

    case 0: s->pict_type= AV_PICTURE_TYPE_I; break;

    case 1: s->pict_type= AV_PICTURE_TYPE_I; break; //hmm ...

    case 2: s->pict_type= AV_PICTURE_TYPE_P; break;

    case 3: s->pict_type= AV_PICTURE_TYPE_B; break;

    default:

        av_log(s->avctx, AV_LOG_ERROR, "unknown frame type\n");

        return -1;

    }



    if(s->low_delay && s->pict_type==AV_PICTURE_TYPE_B){

        av_log(s->avctx, AV_LOG_ERROR, "low delay B\n");

        return -1;

    }

    if(s->last_picture_ptr==NULL && s->pict_type==AV_PICTURE_TYPE_B){

        av_log(s->avctx, AV_LOG_ERROR, "early B pix\n");

        return -1;

    }



    if (get_bits1(&s->gb)){

        av_log(s->avctx, AV_LOG_ERROR, "reserved bit set\n");

        return -1;

    }



    s->qscale = get_bits(&s->gb, 5);

    if(s->qscale==0){

        av_log(s->avctx, AV_LOG_ERROR, "error, qscale:0\n");

        return -1;

    }



    if(RV_GET_MINOR_VER(rv->sub_id) >= 2)

        s->loop_filter = get_bits1(&s->gb) && !s->avctx->lowres;



    if(RV_GET_MINOR_VER(rv->sub_id) <= 1)

        seq = get_bits(&s->gb, 8) << 7;

    else

        seq = get_bits(&s->gb, 13) << 2;



    rpr_bits = s->avctx->extradata[1] & 7;

    if(rpr_bits){

        int f, new_w, new_h;

        rpr_bits = FFMIN((rpr_bits >> 1) + 1, 3);



        f = get_bits(&s->gb, rpr_bits);



        if(f){

            new_w= 4*((uint8_t*)s->avctx->extradata)[6+2*f];

            new_h= 4*((uint8_t*)s->avctx->extradata)[7+2*f];

        }else{

            new_w= s->orig_width ;

            new_h= s->orig_height;

        }

        if(new_w != s->width || new_h != s->height){

            AVRational old_aspect = s->avctx->sample_aspect_ratio;

            av_log(s->avctx, AV_LOG_DEBUG, "attempting to change resolution to %dx%d\n", new_w, new_h);

            if (av_image_check_size(new_w, new_h, 0, s->avctx) < 0)

                return -1;

            ff_MPV_common_end(s);



            // attempt to keep aspect during typical resolution switches

            if (!old_aspect.num)

                old_aspect = (AVRational){1, 1};

            if (2 * new_w * s->height == new_h * s->width)

                s->avctx->sample_aspect_ratio = av_mul_q(old_aspect, (AVRational){2, 1});

            if (new_w * s->height == 2 * new_h * s->width)

                s->avctx->sample_aspect_ratio = av_mul_q(old_aspect, (AVRational){1, 2});

            avcodec_set_dimensions(s->avctx, new_w, new_h);

            s->width  = new_w;

            s->height = new_h;

            if (ff_MPV_common_init(s) < 0)

                return -1;

        }



        if(s->avctx->debug & FF_DEBUG_PICT_INFO){

            av_log(s->avctx, AV_LOG_DEBUG, "F %d/%d\n", f, rpr_bits);

        }

    }

    if (av_image_check_size(s->width, s->height, 0, s->avctx) < 0)

        return AVERROR_INVALIDDATA;



    mb_pos = ff_h263_decode_mba(s);



    seq |= s->time &~0x7FFF;

    if(seq - s->time >  0x4000) seq -= 0x8000;

    if(seq - s->time < -0x4000) seq += 0x8000;

    if(seq != s->time){

        if(s->pict_type!=AV_PICTURE_TYPE_B){

            s->time= seq;

            s->pp_time= s->time - s->last_non_b_time;

            s->last_non_b_time= s->time;

        }else{

            s->time= seq;

            s->pb_time= s->pp_time - (s->last_non_b_time - s->time);

            if(s->pp_time <=s->pb_time || s->pp_time <= s->pp_time - s->pb_time || s->pp_time<=0){

                av_log(s->avctx, AV_LOG_DEBUG, "messed up order, possible from seeking? skipping current b frame\n");

                return FRAME_SKIPPED;

            }

            ff_mpeg4_init_direct_mv(s);

        }

    }



    s->no_rounding= get_bits1(&s->gb);



    if(RV_GET_MINOR_VER(rv->sub_id) <= 1 && s->pict_type == AV_PICTURE_TYPE_B)

        skip_bits(&s->gb, 5); // binary decoder reads 3+2 bits here but they don't seem to be used



    s->f_code = 1;

    s->unrestricted_mv = 1;

    s->h263_aic= s->pict_type == AV_PICTURE_TYPE_I;

//    s->alt_inter_vlc=1;

//    s->obmc=1;

//    s->umvplus=1;

    s->modified_quant=1;

    if(!s->avctx->lowres)

        s->loop_filter=1;



    if(s->avctx->debug & FF_DEBUG_PICT_INFO){

            av_log(s->avctx, AV_LOG_INFO, "num:%5d x:%2d y:%2d type:%d qscale:%2d rnd:%d\n",

                   seq, s->mb_x, s->mb_y, s->pict_type, s->qscale, s->no_rounding);

    }



    av_assert0(s->pict_type != AV_PICTURE_TYPE_B || !s->low_delay);



    return s->mb_width*s->mb_height - mb_pos;

}
