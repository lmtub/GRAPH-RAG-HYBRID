static int load_input_picture(MpegEncContext *s, AVFrame *pic_arg){

    AVFrame *pic=NULL;

    int i;

    const int encoding_delay= s->max_b_frames;

    int direct=1;

    

  if(pic_arg){

    if(encoding_delay && !(s->flags&CODEC_FLAG_INPUT_PRESERVED)) direct=0;

    if(pic_arg->linesize[0] != s->linesize) direct=0;

    if(pic_arg->linesize[1] != s->uvlinesize) direct=0;

    if(pic_arg->linesize[2] != s->uvlinesize) direct=0;

  

//    printf("%d %d %d %d\n",pic_arg->linesize[0], pic_arg->linesize[1], s->linesize, s->uvlinesize);

    

    if(direct){

        i= find_unused_picture(s, 1);



        pic= (AVFrame*)&s->picture[i];

        pic->reference= 3;

    

        for(i=0; i<4; i++){

            pic->data[i]= pic_arg->data[i];

            pic->linesize[i]= pic_arg->linesize[i];

        }

        alloc_picture(s, (Picture*)pic, 1);

    }else{

        i= find_unused_picture(s, 0);



        pic= (AVFrame*)&s->picture[i];

        pic->reference= 3;



        alloc_picture(s, (Picture*)pic, 0);

        for(i=0; i<4; i++){

            /* the input will be 16 pixels to the right relative to the actual buffer start

             * and the current_pic, so the buffer can be reused, yes its not beatifull 

             */

            pic->data[i]+= 16; 

        }



        if(   pic->data[0] == pic_arg->data[0] 

           && pic->data[1] == pic_arg->data[1]

           && pic->data[2] == pic_arg->data[2]){

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

    pic->quality= pic_arg->quality;

    pic->pict_type= pic_arg->pict_type;

    pic->pts = pic_arg->pts;

    

    if(s->input_picture[encoding_delay])

        pic->display_picture_number= s->input_picture[encoding_delay]->display_picture_number + 1;

    

  }



    /* shift buffer entries */

    for(i=1; i<MAX_PICTURE_COUNT /*s->encoding_delay+1*/; i++)

        s->input_picture[i-1]= s->input_picture[i];

        

    s->input_picture[encoding_delay]= (Picture*)pic;



    return 0;

}
