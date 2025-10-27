static int vc1_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            const uint8_t *buf, int buf_size)

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



    /* We need to set current_picture_ptr before reading the header,

     * otherwise we cannot store anything in there. */

    if(s->current_picture_ptr==NULL || s->current_picture_ptr->data[0]){

        int i= ff_find_unused_picture(s, 0);

        s->current_picture_ptr= &s->picture[i];

    }



    //for advanced profile we may need to parse and unescape data

    if (avctx->codec_id == CODEC_ID_VC1) {

        int buf_size2 = 0;

        buf2 = av_mallocz(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);



        if(IS_MARKER(AV_RB32(buf))){ /* frame starts with marker and needs to be parsed */

            const uint8_t *start, *end, *next;

            int size;



            next = buf;

            for(start = buf, end = buf + buf_size; next < end; start = next){

                next = find_next_marker(start + 4, end);

                size = next - start - 4;

                if(size <= 0) continue;

                switch(AV_RB32(start)){

                case VC1_CODE_FRAME:

                    buf_size2 = vc1_unescape_buffer(start + 4, size, buf2);

                    break;

                case VC1_CODE_ENTRYPOINT: /* it should be before frame data */

                    buf_size2 = vc1_unescape_buffer(start + 4, size, buf2);

                    init_get_bits(&s->gb, buf2, buf_size2*8);

                    decode_entry_point(avctx, &s->gb);

                    break;

                case VC1_CODE_SLICE:

                    av_log(avctx, AV_LOG_ERROR, "Sliced decoding is not implemented (yet)\n");


                    return -1;

                }

            }

        }else if(v->interlace && ((buf[0] & 0xC0) == 0xC0)){ /* WVC1 interlaced stores both fields divided by marker */

            const uint8_t *divider;



            divider = find_next_marker(buf, buf + buf_size);

            if((divider == (buf + buf_size)) || AV_RB32(divider) != VC1_CODE_FIELD){

                av_log(avctx, AV_LOG_ERROR, "Error in WVC1 interlaced frame\n");


                return -1;

            }



            buf_size2 = vc1_unescape_buffer(buf, divider - buf, buf2);

            // TODO

            av_free(buf2);return -1;

        }else{

            buf_size2 = vc1_unescape_buffer(buf, buf_size, buf2);

        }

        init_get_bits(&s->gb, buf2, buf_size2*8);

    } else

        init_get_bits(&s->gb, buf, buf_size*8);

    // do parse frame header

    if(v->profile < PROFILE_ADVANCED) {

        if(vc1_parse_frame_header(v, &s->gb) == -1) {


            return -1;

        }

    } else {

        if(vc1_parse_frame_header_adv(v, &s->gb) == -1) {


            return -1;

        }

    }



    if(s->pict_type != FF_I_TYPE && !v->res_rtm_flag){


        return -1;

    }



    // for hurry_up==5

    s->current_picture.pict_type= s->pict_type;

    s->current_picture.key_frame= s->pict_type == FF_I_TYPE;



    /* skip B-frames if we don't have reference frames */

    if(s->last_picture_ptr==NULL && (s->pict_type==FF_B_TYPE || s->dropable)){


        return -1;//buf_size;

    }

    /* skip b frames if we are in a hurry */

    if(avctx->hurry_up && s->pict_type==FF_B_TYPE) return -1;//buf_size;

    if(   (avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type==FF_B_TYPE)

       || (avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type!=FF_I_TYPE)

       ||  avctx->skip_frame >= AVDISCARD_ALL) {


        return buf_size;

    }

    /* skip everything if we are in a hurry>=5 */

    if(avctx->hurry_up>=5) {


        return -1;//buf_size;

    }



    if(s->next_p_frame_damaged){

        if(s->pict_type==FF_B_TYPE)

            return buf_size;

        else

            s->next_p_frame_damaged=0;

    }



    if(MPV_frame_start(s, avctx) < 0) {


        return -1;

    }



    s->me.qpel_put= s->dsp.put_qpel_pixels_tab;

    s->me.qpel_avg= s->dsp.avg_qpel_pixels_tab;



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

    if (s->pict_type == FF_B_TYPE || s->low_delay) {

        *pict= *(AVFrame*)s->current_picture_ptr;

    } else if (s->last_picture_ptr != NULL) {

        *pict= *(AVFrame*)s->last_picture_ptr;

    }



    if(s->last_picture_ptr || s->low_delay){

        *data_size = sizeof(AVFrame);

        ff_print_debug_info(s, pict);

    }



    /* Return the Picture timestamp as the frame number */

    /* we subtract 1 because it is added on utils.c     */

    avctx->frame_number = s->picture_number - 1;




    return buf_size;

}