static int decode_frame(AVCodecContext *avctx, 

                        void *data, int *data_size,

                        uint8_t *buf, int buf_size)

{

    MDECContext * const a = avctx->priv_data;

    AVFrame *picture = data;

    AVFrame * const p= (AVFrame*)&a->picture;

    int i;



    /* special case for last picture */

    if (buf_size == 0) {

        return 0;

    }



    if(p->data[0])

        avctx->release_buffer(avctx, p);



    p->reference= 0;

    if(avctx->get_buffer(avctx, p) < 0){

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }

    p->pict_type= I_TYPE;

    p->key_frame= 1;

    a->last_dc[0]=

    a->last_dc[1]=

    a->last_dc[2]= 0;



    a->bitstream_buffer= av_fast_realloc(a->bitstream_buffer, &a->bitstream_buffer_size, buf_size + FF_INPUT_BUFFER_PADDING_SIZE);

    for(i=0; i<buf_size; i+=2){

        a->bitstream_buffer[i]  = buf[i+1];

        a->bitstream_buffer[i+1]= buf[i  ];

    }

    init_get_bits(&a->gb, a->bitstream_buffer, buf_size*8);

    

    /* skip over 4 preamble bytes in stream (typically 0xXX 0xXX 0x00 0x38) */

    skip_bits(&a->gb, 32);



    a->qscale=  get_bits(&a->gb, 16);

    a->version= get_bits(&a->gb, 16);

    

//    printf("qscale:%d (0x%X), version:%d (0x%X)\n", a->qscale, a->qscale, a->version, a->version);

    

    for(a->mb_x=0; a->mb_x<a->mb_width; a->mb_x++){

        for(a->mb_y=0; a->mb_y<a->mb_height; a->mb_y++){

            if( decode_mb(a, a->block) <0)

                return -1;

             

            idct_put(a, a->mb_x, a->mb_y);

        }

    }



//    p->quality= (32 + a->inv_qscale/2)/a->inv_qscale;

//    memset(p->qscale_table, p->quality, p->qstride*a->mb_height);

    

    *picture= *(AVFrame*)&a->picture;

    *data_size = sizeof(AVPicture);



    emms_c();

    

    return (get_bits_count(&a->gb)+31)/32*4;

}
