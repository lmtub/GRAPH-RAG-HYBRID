static int mpeg_decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             UINT8 *buf, int buf_size)

{

    Mpeg1Context *s = avctx->priv_data;

    UINT8 *buf_end, *buf_ptr, *buf_start;

    int len, start_code_found, ret, code, start_code, input_size;

    AVPicture *picture = data;

    MpegEncContext *s2 = &s->mpeg_enc_ctx;

            

    dprintf("fill_buffer\n");



    *data_size = 0;

    

    /* special case for last picture */

    if (buf_size == 0) {

        if (s2->picture_number > 0) {

            picture->data[0] = s2->next_picture[0];

            picture->data[1] = s2->next_picture[1];

            picture->data[2] = s2->next_picture[2];

            picture->linesize[0] = s2->linesize;

            picture->linesize[1] = s2->linesize / 2;

            picture->linesize[2] = s2->linesize / 2;

            *data_size = sizeof(AVPicture);

        }

        return 0;

    }



    buf_ptr = buf;

    buf_end = buf + buf_size;

    

    if (s->repeat_field % 2 == 1) {

        s->repeat_field++;

        //fprintf(stderr,"\nRepeating last frame: %d -> %d! pict: %d %d", avctx->frame_number-1, avctx->frame_number,

        //                                                         s2->picture_number, s->repeat_field);

        *data_size = sizeof(AVPicture);

        goto the_end;

    }

        

    while (buf_ptr < buf_end) {

        buf_start = buf_ptr;

        /* find start next code */

        code = find_start_code(&buf_ptr, buf_end, &s->header_state);

        if (code >= 0) {

            start_code_found = 1;

        } else {

            start_code_found = 0;

        }

        /* copy to buffer */

        len = buf_ptr - buf_start;

        if (len + (s->buf_ptr - s->buffer) > s->buffer_size) {

            /* data too big : flush */

            s->buf_ptr = s->buffer;

            if (start_code_found)

                s->start_code = code;

        } else {

            memcpy(s->buf_ptr, buf_start, len);

            s->buf_ptr += len;

            

            if (start_code_found) {

                /* prepare data for next start code */

                input_size = s->buf_ptr - s->buffer;

                start_code = s->start_code;

                s->buf_ptr = s->buffer;

                s->start_code = code;

                switch(start_code) {

                case SEQ_START_CODE:

                    mpeg1_decode_sequence(avctx, s->buffer, 

                                          input_size);

                    break;

                            

                case PICTURE_START_CODE:

                    /* we have a complete image : we try to decompress it */

                    mpeg1_decode_picture(avctx, 

                                         s->buffer, input_size);

                    break;

                case EXT_START_CODE:

                    mpeg_decode_extension(avctx,

                                          s->buffer, input_size);

                    break;

                default:

                    if (start_code >= SLICE_MIN_START_CODE &&

                        start_code <= SLICE_MAX_START_CODE) {

                        ret = mpeg_decode_slice(avctx, picture,

                                                start_code, s->buffer, input_size);

                        if (ret == 1) {

                            /* got a picture: exit */

                            /* first check if we must repeat the frame */

                            if (s2->progressive_frame && s2->repeat_first_field) {

                                //fprintf(stderr,"\nRepeat this frame: %d! pict: %d",avctx->frame_number,s2->picture_number);

                                s2->repeat_first_field = 0;

                                s2->progressive_frame = 0;

                                if (++s->repeat_field > 2)

                                    s->repeat_field = 0;

                            }

                            *data_size = sizeof(AVPicture);

                            goto the_end;

                        }

                    }

                    break;

                }

            }

        }

    }

 the_end:

    return buf_ptr - buf;

}
