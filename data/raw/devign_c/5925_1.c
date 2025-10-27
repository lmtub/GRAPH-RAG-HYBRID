static int svq1_encode_frame(AVCodecContext *avctx, unsigned char *buf,

    int buf_size, void *data)

{

    SVQ1Context * const s = avctx->priv_data;

    AVFrame *pict = data;

    AVFrame * const p= (AVFrame*)&s->picture;

    AVFrame temp;

    int i;



    if(avctx->pix_fmt != PIX_FMT_YUV410P){

        av_log(avctx, AV_LOG_ERROR, "unsupported pixel format\n");

        return -1;

    }



    if(!s->current_picture.data[0]){

        avctx->get_buffer(avctx, &s->current_picture);

        avctx->get_buffer(avctx, &s->last_picture);

        s->scratchbuf = av_malloc(s->current_picture.linesize[0] * 16);

    }



    temp= s->current_picture;

    s->current_picture= s->last_picture;

    s->last_picture= temp;



    init_put_bits(&s->pb, buf, buf_size);



    *p = *pict;

    p->pict_type = avctx->gop_size && avctx->frame_number % avctx->gop_size ? FF_P_TYPE : FF_I_TYPE;

    p->key_frame = p->pict_type == FF_I_TYPE;



    svq1_write_header(s, p->pict_type);

    for(i=0; i<3; i++){

        if(svq1_encode_plane(s, i,

            s->picture.data[i], s->last_picture.data[i], s->current_picture.data[i],

            s->frame_width / (i?4:1), s->frame_height / (i?4:1),

            s->picture.linesize[i], s->current_picture.linesize[i]) < 0)

                return -1;

    }



//    align_put_bits(&s->pb);

    while(put_bits_count(&s->pb) & 31)

        put_bits(&s->pb, 1, 0);



    flush_put_bits(&s->pb);



    return put_bits_count(&s->pb) / 8;

}
