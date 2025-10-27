static int load_input_picture(MpegEncContext *s, AVFrame *pic_arg){

    AVFrame *pic=NULL;

    int64_t pts;

    int i;

    const int encoding_delay= s->max_b_frames;

    int direct=1;



    if(pic_arg){

        pts= pic_arg->pts;

        pic_arg->display_picture_number= s->input_picture_number++;



        if(pts != AV_NOPTS_VALUE){

            if(s->user_specified_pts != AV_NOPTS_VALUE){

                int64_t time= pts;

                int64_t last= s->user_specified_pts;



                if(time <= last){

                    av_log(s->avctx, AV_LOG_ERROR, "Error, Invalid timestamp=%"PRId64", last=%"PRId64"\n", pts, s->user_specified_pts);

                    return -1;

                }

            }

            s->user_specified_pts= pts;

        }else{

            if(s->user_specified_pts != AV_NOPTS_VALUE){

                s->user_specified_pts=

                pts= s->user_specified_pts + 1;

                av_log(s->avctx, AV_LOG_INFO, "Warning: AVFrame.pts=? trying to guess (%"PRId64")\n", pts);

            }else{

                pts= pic_arg->display_picture_number;

            }

        }

    }



  if(pic_arg){

    if(encoding_delay && !(s->flags&CODEC_FLAG_INPUT_PRESERVED)) direct=0;

    if(pic_arg->linesize[0] != s->linesize) direct=0;

    if(pic_arg->linesize[1] != s->uvlinesize) direct=0;

    if(pic_arg->linesize[2] != s->uvlinesize) direct=0;



//    av_log(AV_LOG_DEBUG, "%d %d %d %d\n",pic_arg->linesize[0], pic_arg->linesize[1], s->linesize, s->uvlinesize);



    if(direct){

        i= ff_find_unused_picture(s, 1);



        pic= (AVFrame*)&s->picture[i];

        pic->reference= 3;



        for(i=0; i<4; i++){

            pic->data[i]= pic_arg->data[i];

            pic->linesize[i]= pic_arg->linesize[i];

        }

        ff_alloc_picture(s, (Picture*)pic, 1);

    }else{

        i= ff_find_unused_picture(s, 0);



        pic= (AVFrame*)&s->picture[i];

        pic->reference= 3;



        ff_alloc_picture(s, (Picture*)pic, 0);



        if(   pic->data[0] + INPLACE_OFFSET == pic_arg->data[0]

           && pic->data[1] + INPLACE_OFFSET == pic_arg->data[1]

           && pic->data[2] + INPLACE_OFFSET == pic_arg->data[2]){

       // empty

        }else{

            int h_chroma_shift, v_chroma_shift;

            avcodec_get_chroma_sub_sample(s->avctx->pix_fmt, &h_chroma_shift, &v_chroma_shift);



            for(i=0; i<3; i++){

                int src_stride= pic_arg->linesize[i];

                int dst_stride= i ? s->uvlinesize : s->linesize;

                int h_shift= i ? h_chroma_shift : 0;

                int v_shift= i ? v_chroma_shift : 0;

                int w= s->width >>h_shift;

                int h= s->height>>v_shift;

                uint8_t *src= pic_arg->data[i];

                uint8_t *dst= pic->data[i];



                if(!s->avctx->rc_buffer_size)

                    dst +=INPLACE_OFFSET;



                if(src_stride==dst_stride)

                    memcpy(dst, src, src_stride*h);

                else{

                    while(h--){

                        memcpy(dst, src, w);

                        dst += dst_stride;

                        src += src_stride;

                    }

                }

            }

        }

    }

    copy_picture_attributes(s, pic, pic_arg);

    pic->pts= pts; //we set this here to avoid modifiying pic_arg

  }



    /* shift buffer entries */

    for(i=1; i<MAX_PICTURE_COUNT /*s->encoding_delay+1*/; i++)

        s->input_picture[i-1]= s->input_picture[i];



    s->input_picture[encoding_delay]= (Picture*)pic;



    return 0;

}
