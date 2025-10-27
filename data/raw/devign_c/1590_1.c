static int v410_decode_frame(AVCodecContext *avctx, void *data,
                             int *data_size, AVPacket *avpkt)
{
    AVFrame *pic = avctx->coded_frame;
    uint8_t *src = avpkt->data;
    uint16_t *y, *u, *v;
    uint32_t val;
    int i, j;
    if (pic->data[0])
        avctx->release_buffer(avctx, pic);
    pic->reference = 0;
    if (avctx->get_buffer(avctx, pic) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Could not allocate buffer.\n");
        return AVERROR(ENOMEM);
    pic->key_frame = 1;
    pic->pict_type = FF_I_TYPE;
    y = (uint16_t *)pic->data[0];
    u = (uint16_t *)pic->data[1];
    v = (uint16_t *)pic->data[2];
    for (i = 0; i < avctx->height; i++) {
        for (j = 0; j < avctx->width; j++) {
            val = AV_RL32(src);
            u[j] = (val >>  2) & 0x3FF;
            y[j] = (val >> 12) & 0x3FF;
            v[j] = (val >> 22);
            src += 4;
        y += pic->linesize[0] >> 1;
        u += pic->linesize[1] >> 1;
        v += pic->linesize[2] >> 1;
    *data_size = sizeof(AVFrame);
    *(AVFrame *)data = *pic;
    return avpkt->size;