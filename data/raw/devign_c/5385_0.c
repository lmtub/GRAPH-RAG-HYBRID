static int vc1_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    VC1Context *v = avctx->priv_data;

    MpegEncContext *s = &v->s;

    AVFrame *pict = data;

    uint8_t *buf2 = NULL;



    /* no supplementary picture */

    if (buf_size == 0) {

        /* special case for last picture */

        if (s->low_delay==0 && s->next_picture_ptr) {

            *pict= *(AVFrame*)s->next_picture_ptr;

            s->next_picture_ptr= NULL;



            *data_size = sizeof(AVFrame);

        }



        return 0;

    }



    //we need to set current_picture_ptr before reading the header, otherwise we cant store anyting im there

    if(s->current_picture_ptr==NULL || s->current_picture_ptr->data[0]){

        int i= ff_find_unused_picture(s, 0);

        s->current_picture_ptr= &s->picture[i];

    }



    avctx->has_b_frames= !s->low_delay;



    //for advanced profile we need to unescape buffer

    if (avctx->codec_id == CODEC_ID_VC1) {

        int i, buf_size2;

        buf2 = av_malloc(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);

        buf_size2 = 0;

        for(i = 0; i < buf_size; i++) {

            if(buf[i] == 3 && i >= 2 && !buf[i-1] && !buf[i-2] && i < buf_size-1 && buf[i+1] < 4) {

                buf2[buf_size2++] = buf[i+1];

                i++;

            } else

                buf2[buf_size2++] = buf[i];

        }

        init_get_bits(&s->gb, buf2, buf_size2*8);

    } else

        init_get_bits(&s->gb, buf, buf_size*8);

    // do parse frame header

    if(v->profile < PROFILE_ADVANCED) {

        if(vc1_parse_frame_header(v, &s->gb) == -1) {

            if(buf2)av_free(buf2);

            return -1;

        }

    } else {

        if(vc1_parse_frame_header_adv(v, &s->gb) == -1) {

            if(buf2)av_free(buf2);

            return -1;

        }

    }



    if(s->pict_type != I_TYPE && !v->res_rtm_flag){

        if(buf2)av_free(buf2);

        return -1;

    }



    // for hurry_up==5

    s->current_picture.pict_type= s->pict_type;

    s->current_picture.key_frame= s->pict_type == I_TYPE;



    /* skip B-frames if we don't have reference frames */

    if(s->last_picture_ptr==NULL && (s->pict_type==B_TYPE || s->dropable)){

        if(buf2)av_free(buf2);

        return -1;//buf_size;

    }

    /* skip b frames if we are in a hurry */

    if(avctx->hurry_up && s->pict_type==B_TYPE) return -1;//buf_size;

    if(   (avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type==B_TYPE)

       || (avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type!=I_TYPE)

       ||  avctx->skip_frame >= AVDISCARD_ALL) {

        if(buf2)av_free(buf2);

        return buf_size;

    }

    /* skip everything if we are in a hurry>=5 */

    if(avctx->hurry_up>=5) {

        if(buf2)av_free(buf2);

        return -1;//buf_size;

    }



    if(s->next_p_frame_damaged){

        if(s->pict_type==B_TYPE)

            return buf_size;

        else

            s->next_p_frame_damaged=0;

    }



    if(MPV_frame_start(s, avctx) < 0) {

        if(buf2)av_free(buf2);

        return -1;

    }



    ff_er_frame_start(s);



    v->bits = buf_size * 8;

    vc1_decode_blocks(v);

//av_log(s->avctx, AV_LOG_INFO, "Consumed %i/%i bits\n", get_bits_count(&s->gb), buf_size*8);

//  if(get_bits_count(&s->gb) > buf_size * 8)

//      return -1;

    ff_er_frame_end(s);



    MPV_frame_end(s);



assert(s->current_picture.pict_type == s->current_picture_ptr->pict_type);

assert(s->current_picture.pict_type == s->pict_type);

    if (s->pict_type == B_TYPE || s->low_delay) {

        *pict= *(AVFrame*)s->current_picture_ptr;

    } else if (s->last_picture_ptr != NULL) {

        *pict= *(AVFrame*)s->last_picture_ptr;

    }



    if(s->last_picture_ptr || s->low_delay){

        *data_size = sizeof(AVFrame);

        ff_print_debug_info(s, pict);

    }



    /* Return the Picture timestamp as the frame number */

    /* we substract 1 because it is added on utils.c    */

    avctx->frame_number = s->picture_number - 1;



    if(buf2)av_free(buf2);

    return buf_size;

}
