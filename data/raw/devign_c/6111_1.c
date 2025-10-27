static int rv20_decode_picture_header(MpegEncContext *s)

{

    int seq, mb_pos, i;

    

    if(s->avctx->sub_id == 0x30202002 || s->avctx->sub_id == 0x30203002){

        if (get_bits(&s->gb, 3)){

            av_log(s->avctx, AV_LOG_ERROR, "unknown triplet set\n");

            return -1;

        } 

    }   



    i= get_bits(&s->gb, 2);

    switch(i){

    case 0: s->pict_type= I_TYPE; break;

    case 1: s->pict_type= I_TYPE; break; //hmm ...

    case 2: s->pict_type= P_TYPE; break;

    case 3: s->pict_type= B_TYPE; break;

    default: 

        av_log(s->avctx, AV_LOG_ERROR, "unknown frame type\n");

        return -1;

    }

    

    if (get_bits(&s->gb, 1)){

        av_log(s->avctx, AV_LOG_ERROR, "unknown bit set\n");

        return -1;

    }



    s->qscale = get_bits(&s->gb, 5);

    if(s->qscale==0){

        av_log(s->avctx, AV_LOG_ERROR, "error, qscale:0\n");

        return -1;

    }

    if(s->avctx->sub_id == 0x30203002){

        if (get_bits(&s->gb, 1)){

            av_log(s->avctx, AV_LOG_ERROR, "unknown bit2 set\n");

            return -1;

        }

    }

        

    if(s->avctx->has_b_frames){

        if (get_bits(&s->gb, 1)){

            av_log(s->avctx, AV_LOG_ERROR, "unknown bit3 set\n");

            return -1;

        }

        seq= get_bits(&s->gb, 15);

    }else

        seq= get_bits(&s->gb, 8)*128;

//printf("%d\n", seq);

    seq |= s->time &~0x7FFF;

    if(seq - s->time >  0x4000) seq -= 0x8000;

    if(seq - s->time < -0x4000) seq += 0x8000;

    if(seq != s->time){  

        if(s->pict_type!=B_TYPE){

            s->time= seq;

            s->pp_time= s->time - s->last_non_b_time;

            s->last_non_b_time= s->time;

        }else{

            s->time= seq;

            s->pb_time= s->pp_time - (s->last_non_b_time - s->time);

            if(s->pp_time <=s->pb_time || s->pp_time <= s->pp_time - s->pb_time || s->pp_time<=0){

                printf("messed up order, seeking?, skiping current b frame\n");

                return FRAME_SKIPED;

            }

        }

    }

//    printf("%d %d %d %d %d\n", seq, (int)s->time, (int)s->last_non_b_time, s->pp_time, s->pb_time);



    mb_pos= ff_h263_decode_mba(s);

    s->no_rounding= get_bits1(&s->gb);

    

    s->f_code = 1;

    s->unrestricted_mv = 1;

    s->h263_aic= s->pict_type == I_TYPE;

//    s->alt_inter_vlc=1;

//    s->obmc=1;

//    s->umvplus=1;

    s->modified_quant=1;

    s->loop_filter=1;

    

    if(s->avctx->debug & FF_DEBUG_PICT_INFO){

            av_log(s->avctx, AV_LOG_INFO, "num:%5d x:%2d y:%2d type:%d qscale:%2d rnd:%d\n", 

                   seq, s->mb_x, s->mb_y, s->pict_type, s->qscale, s->no_rounding);

    }



    assert(s->pict_type != B_TYPE || !s->low_delay);



    return s->mb_width*s->mb_height - mb_pos;

}
