static int ljpeg_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                              const AVFrame *pict, int *got_packet)

{

    LJpegEncContext *s = avctx->priv_data;

    PutBitContext pb;

    const int width  = avctx->width;

    const int height = avctx->height;

    const int mb_width  = (width  + s->hsample[0] - 1) / s->hsample[0];

    const int mb_height = (height + s->vsample[0] - 1) / s->vsample[0];

    int max_pkt_size = AV_INPUT_BUFFER_MIN_SIZE;

    int ret, header_bits;



    if (avctx->pix_fmt == AV_PIX_FMT_BGR24)

        max_pkt_size += width * height * 3 * 3;

    else {

        max_pkt_size += mb_width * mb_height * 3 * 4

                        * s->hsample[0] * s->vsample[0];

    }

    if ((ret = ff_alloc_packet(pkt, max_pkt_size)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet of size %d.\n", max_pkt_size);

        return ret;

    }



    init_put_bits(&pb, pkt->data, pkt->size);



    ff_mjpeg_encode_picture_header(avctx, &pb, &s->scantable,

                                   s->matrix);



    header_bits = put_bits_count(&pb);



    if (avctx->pix_fmt == AV_PIX_FMT_BGR24)

        ret = ljpeg_encode_bgr(avctx, &pb, pict);

    else

        ret = ljpeg_encode_yuv(avctx, &pb, pict);

    if (ret < 0)

        return ret;



    emms_c();



    ff_mjpeg_encode_picture_trailer(&pb, header_bits);



    flush_put_bits(&pb);

    pkt->size   = put_bits_ptr(&pb) - pb.buf;

    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}
