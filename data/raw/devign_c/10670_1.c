int alloc_picture(MpegEncContext *s, Picture *pic, int shared){

    const int big_mb_num= s->mb_stride*(s->mb_height+1) + 1; //the +1 is needed so memset(,,stride*height) does not sig11

    const int mb_array_size= s->mb_stride*s->mb_height;

    const int b8_array_size= s->b8_stride*s->mb_height*2;

    const int b4_array_size= s->b4_stride*s->mb_height*4;

    int i;



    if(shared){

        assert(pic->data[0]);

        assert(pic->type == 0 || pic->type == FF_BUFFER_TYPE_SHARED);

        pic->type= FF_BUFFER_TYPE_SHARED;

    }else{

        int r;



        assert(!pic->data[0]);



        r= s->avctx->get_buffer(s->avctx, (AVFrame*)pic);



        if(r<0 || !pic->age || !pic->type || !pic->data[0]){

            av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed (%d %d %d %p)\n", r, pic->age, pic->type, pic->data[0]);

            return -1;

        }



        if(s->linesize && (s->linesize != pic->linesize[0] || s->uvlinesize != pic->linesize[1])){

            av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed (stride changed)\n");

            return -1;

        }



        if(pic->linesize[1] != pic->linesize[2]){

            av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed (uv stride mismatch)\n");

            return -1;

        }



        s->linesize  = pic->linesize[0];

        s->uvlinesize= pic->linesize[1];

    }



    if(pic->qscale_table==NULL){

        if (s->encoding) {

            CHECKED_ALLOCZ(pic->mb_var   , mb_array_size * sizeof(int16_t))

            CHECKED_ALLOCZ(pic->mc_mb_var, mb_array_size * sizeof(int16_t))

            CHECKED_ALLOCZ(pic->mb_mean  , mb_array_size * sizeof(int8_t))

        }



        CHECKED_ALLOCZ(pic->mbskip_table , mb_array_size * sizeof(uint8_t)+2) //the +2 is for the slice end check

        CHECKED_ALLOCZ(pic->qscale_table , mb_array_size * sizeof(uint8_t))

        CHECKED_ALLOCZ(pic->mb_type_base , big_mb_num    * sizeof(uint32_t))

        pic->mb_type= pic->mb_type_base + s->mb_stride+1;

        if(s->out_format == FMT_H264){

            for(i=0; i<2; i++){

                CHECKED_ALLOCZ(pic->motion_val_base[i], 2 * (b4_array_size+4)  * sizeof(int16_t))

                pic->motion_val[i]= pic->motion_val_base[i]+4;

                CHECKED_ALLOCZ(pic->ref_index[i], b8_array_size * sizeof(uint8_t))

            }

            pic->motion_subsample_log2= 2;

        }else if(s->out_format == FMT_H263 || s->encoding || (s->avctx->debug&FF_DEBUG_MV) || (s->avctx->debug_mv)){

            for(i=0; i<2; i++){

                CHECKED_ALLOCZ(pic->motion_val_base[i], 2 * (b8_array_size+4) * sizeof(int16_t))

                pic->motion_val[i]= pic->motion_val_base[i]+4;

                CHECKED_ALLOCZ(pic->ref_index[i], b8_array_size * sizeof(uint8_t))

            }

            pic->motion_subsample_log2= 3;

        }

        if(s->avctx->debug&FF_DEBUG_DCT_COEFF) {

            CHECKED_ALLOCZ(pic->dct_coeff, 64 * mb_array_size * sizeof(DCTELEM)*6)

        }

        pic->qstride= s->mb_stride;

        CHECKED_ALLOCZ(pic->pan_scan , 1 * sizeof(AVPanScan))

    }



    /* It might be nicer if the application would keep track of these

     * but it would require an API change. */

    memmove(s->prev_pict_types+1, s->prev_pict_types, PREV_PICT_TYPES_BUFFER_SIZE-1);

    s->prev_pict_types[0]= s->pict_type;

    if(pic->age < PREV_PICT_TYPES_BUFFER_SIZE && s->prev_pict_types[pic->age] == B_TYPE)

        pic->age= INT_MAX; // Skipped MBs in B-frames are quite rare in MPEG-1/2 and it is a bit tricky to skip them anyway.



    return 0;

fail: //for the CHECKED_ALLOCZ macro

    return -1;

}
