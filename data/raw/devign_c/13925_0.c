static int svq1_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                             const AVFrame *pict, int *got_packet)

{

    SVQ1EncContext *const s = avctx->priv_data;

    AVFrame *const p        = avctx->coded_frame;

    int i, ret;



    if (!pkt->data &&

        (ret = av_new_packet(pkt, s->y_block_width * s->y_block_height *

                             MAX_MB_BYTES * 3 + FF_MIN_BUFFER_SIZE)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet.\n");

        return ret;

    }



    if (avctx->pix_fmt != AV_PIX_FMT_YUV410P) {

        av_log(avctx, AV_LOG_ERROR, "unsupported pixel format\n");

        return -1;

    }



    if (!s->current_picture->data[0]) {

        ret = ff_get_buffer(avctx, s->current_picture, 0);

        if (ret < 0)

            return ret;

    }

    if (!s->last_picture->data[0]) {

        ret = ff_get_buffer(avctx, s->last_picture, 0);

        if (ret < 0)

            return ret;

    }

    if (!s->scratchbuf) {

        s->scratchbuf = av_malloc(s->current_picture->linesize[0] * 16 * 2);

        if (!s->scratchbuf)

            return AVERROR(ENOMEM);

    }



    FFSWAP(AVFrame*, s->current_picture, s->last_picture);



    init_put_bits(&s->pb, pkt->data, pkt->size);



    p->pict_type = avctx->gop_size && avctx->frame_number % avctx->gop_size ?

                   AV_PICTURE_TYPE_P : AV_PICTURE_TYPE_I;

    p->key_frame = p->pict_type == AV_PICTURE_TYPE_I;

    p->quality   = pict->quality;



    svq1_write_header(s, p->pict_type);

    for (i = 0; i < 3; i++)

        if (svq1_encode_plane(s, i,

                              pict->data[i],

                              s->last_picture->data[i],

                              s->current_picture->data[i],

                              s->frame_width  / (i ? 4 : 1),

                              s->frame_height / (i ? 4 : 1),

                              pict->linesize[i],

                              s->current_picture->linesize[i]) < 0)

            return -1;



    // avpriv_align_put_bits(&s->pb);

    while (put_bits_count(&s->pb) & 31)

        put_bits(&s->pb, 1, 0);



    flush_put_bits(&s->pb);



    pkt->size = put_bits_count(&s->pb) / 8;

    if (p->pict_type == AV_PICTURE_TYPE_I)

        pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}
