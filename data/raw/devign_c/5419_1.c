static int libx265_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                                const AVFrame *pic, int *got_packet)

{

    libx265Context *ctx = avctx->priv_data;

    x265_picture x265pic;

    x265_picture x265pic_out = { { 0 } };

    x265_nal *nal;

    uint8_t *dst;

    int payload = 0;

    int nnal;

    int ret;

    int i;



    x265_picture_init(ctx->params, &x265pic);



    if (pic) {

        for (i = 0; i < 3; i++) {

           x265pic.planes[i] = pic->data[i];

           x265pic.stride[i] = pic->linesize[i];

        }



        x265pic.pts      = pic->pts;

        x265pic.bitDepth = av_pix_fmt_desc_get(avctx->pix_fmt)->comp[0].depth_minus1 + 1;



        x265pic.sliceType = pic->pict_type == AV_PICTURE_TYPE_I ? X265_TYPE_I :

                            pic->pict_type == AV_PICTURE_TYPE_P ? X265_TYPE_P :

                            pic->pict_type == AV_PICTURE_TYPE_B ? X265_TYPE_B :

                            X265_TYPE_AUTO;

    }



    ret = x265_encoder_encode(ctx->encoder, &nal, &nnal,

                              pic ? &x265pic : NULL, &x265pic_out);

    if (ret < 0)

        return AVERROR_UNKNOWN;



    if (!nnal)

        return 0;



    for (i = 0; i < nnal; i++)

        payload += nal[i].sizeBytes;



    ret = ff_alloc_packet(pkt, payload);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet.\n");

        return ret;

    }

    dst = pkt->data;



    for (i = 0; i < nnal; i++) {

        memcpy(dst, nal[i].payload, nal[i].sizeBytes);

        dst += nal[i].sizeBytes;



        if (is_keyframe(nal[i].type))

            pkt->flags |= AV_PKT_FLAG_KEY;

    }



    pkt->pts = x265pic_out.pts;

    pkt->dts = x265pic_out.dts;



    switch (x265pic_out.sliceType) {

    case X265_TYPE_IDR:

    case X265_TYPE_I:

        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

        break;

    case X265_TYPE_P:

        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_P;

        break;

    case X265_TYPE_B:

        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_B;

        break;

    }



    *got_packet = 1;

    return 0;

}
