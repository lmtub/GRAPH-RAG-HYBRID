static int amv_encode_picture(AVCodecContext *avctx, AVPacket *pkt,

                              const AVFrame *pic_arg, int *got_packet)



{

    MpegEncContext *s = avctx->priv_data;

    AVFrame *pic;

    int i, ret;

    int chroma_h_shift, chroma_v_shift;



    av_pix_fmt_get_chroma_sub_sample(avctx->pix_fmt, &chroma_h_shift, &chroma_v_shift);



    //CODEC_FLAG_EMU_EDGE have to be cleared

    if(s->avctx->flags & CODEC_FLAG_EMU_EDGE)

        return AVERROR(EINVAL);



    pic = av_frame_alloc();

    if (!pic)

        return AVERROR(ENOMEM);

    av_frame_ref(pic, pic_arg);

    //picture should be flipped upside-down

    for(i=0; i < 3; i++) {

        int vsample = i ? 2 >> chroma_v_shift : 2;

        pic->data[i] += (pic->linesize[i] * (vsample * (8 * s->mb_height -((s->height/V_MAX)&7)) - 1 ));

        pic->linesize[i] *= -1;

    }

    ret = ff_MPV_encode_picture(avctx, pkt, pic, got_packet);

    av_frame_free(&pic);

    return ret;

}
