static int mpeg_field_start(MpegEncContext *s){

    AVCodecContext *avctx= s->avctx;

    Mpeg1Context *s1 = (Mpeg1Context*)s;



    /* start frame decoding */

    if(s->first_field || s->picture_structure==PICT_FRAME){

        if(MPV_frame_start(s, avctx) < 0)

            return -1;



        ff_er_frame_start(s);



        /* first check if we must repeat the frame */

        s->current_picture_ptr->repeat_pict = 0;

        if (s->repeat_first_field) {

            if (s->progressive_sequence) {

                if (s->top_field_first)

                    s->current_picture_ptr->repeat_pict = 4;

                else

                    s->current_picture_ptr->repeat_pict = 2;

            } else if (s->progressive_frame) {

                s->current_picture_ptr->repeat_pict = 1;

            }

        }



        *s->current_picture_ptr->pan_scan= s1->pan_scan;

    }else{ //second field

            int i;



            if(!s->current_picture_ptr){

                av_log(s->avctx, AV_LOG_ERROR, "first field missing\n");

                return -1;

            }



            for(i=0; i<4; i++){

                s->current_picture.data[i] = s->current_picture_ptr->data[i];

                if(s->picture_structure == PICT_BOTTOM_FIELD){

                    s->current_picture.data[i] += s->current_picture_ptr->linesize[i];

                }

            }

    }

#if CONFIG_MPEG_XVMC_DECODER

// MPV_frame_start will call this function too,

// but we need to call it on every field

    if(s->avctx->xvmc_acceleration)

         ff_xvmc_field_start(s,avctx);

#endif



    return 0;

}
