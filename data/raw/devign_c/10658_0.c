static int h261_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             uint8_t *buf, int buf_size)

{

    H261Context *h= avctx->priv_data;

    MpegEncContext *s = &h->s;

    int ret;

    AVFrame *pict = data;



#ifdef DEBUG

    printf("*****frame %d size=%d\n", avctx->frame_number, buf_size);

    printf("bytes=%x %x %x %x\n", buf[0], buf[1], buf[2], buf[3]);

#endif

    s->flags= avctx->flags;

    s->flags2= avctx->flags2;



    /* no supplementary picture */

    if (buf_size == 0) {

        return 0;

    }

    

    h->gob_start_code_skipped=0;



retry:



    init_get_bits(&s->gb, buf, buf_size*8);



    if(!s->context_initialized){

        if (MPV_common_init(s) < 0) //we need the idct permutaton for reading a custom matrix

            return -1;

    }



    //we need to set current_picture_ptr before reading the header, otherwise we cant store anyting im there

    if(s->current_picture_ptr==NULL || s->current_picture_ptr->data[0]){

        int i= ff_find_unused_picture(s, 0);

        s->current_picture_ptr= &s->picture[i];

    }



    ret = h261_decode_picture_header(h);



    /* skip if the header was thrashed */

    if (ret < 0){

        av_log(s->avctx, AV_LOG_ERROR, "header damaged\n");

        return -1;

    }



    if (s->width != avctx->coded_width || s->height != avctx->coded_height){

        ParseContext pc= s->parse_context; //FIXME move these demuxng hack to avformat

        s->parse_context.buffer=0;

        MPV_common_end(s);

        s->parse_context= pc;

    }

    if (!s->context_initialized) {

        avcodec_set_dimensions(avctx, s->width, s->height);



        goto retry;

    }



    // for hurry_up==5

    s->current_picture.pict_type= s->pict_type;

    s->current_picture.key_frame= s->pict_type == I_TYPE;



    /* skip everything if we are in a hurry>=5 */

    if(avctx->hurry_up>=5) return get_consumed_bytes(s, buf_size);



    if(MPV_frame_start(s, avctx) < 0)

        return -1;



    ff_er_frame_start(s);



    /* decode each macroblock */

    s->mb_x=0;

    s->mb_y=0;



    while(h->gob_number < (s->mb_height==18 ? 12 : 5)){

        if(ff_h261_resync(h)<0)

            break;

        h261_decode_gob(h);

    }

    MPV_frame_end(s);



assert(s->current_picture.pict_type == s->current_picture_ptr->pict_type);

assert(s->current_picture.pict_type == s->pict_type);

    *pict= *(AVFrame*)s->current_picture_ptr;

    ff_print_debug_info(s, pict);



    /* Return the Picture timestamp as the frame number */

    /* we substract 1 because it is added on utils.c    */

    avctx->frame_number = s->picture_number - 1;



    *data_size = sizeof(AVFrame);



    return get_consumed_bytes(s, buf_size);

}
