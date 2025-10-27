static int smvjpeg_decode_frame(AVCodecContext *avctx, void *data, int *data_size,
                            AVPacket *avpkt)
{
    const AVPixFmtDescriptor *desc;
    SMVJpegDecodeContext *s = avctx->priv_data;
    AVFrame* mjpeg_data = s->picture[0];
    int i, cur_frame = 0, ret = 0;
    cur_frame = avpkt->pts % s->frames_per_jpeg;
    /* Are we at the start of a block? */
    if (!cur_frame) {
        av_frame_unref(mjpeg_data);
        ret = avcodec_decode_video2(s->avctx, mjpeg_data, &s->mjpeg_data_size, avpkt);
        if (ret < 0) {
            s->mjpeg_data_size = 0;
            return ret;
        }
    } else if (!s->mjpeg_data_size)
        return AVERROR(EINVAL);
    desc = av_pix_fmt_desc_get(s->avctx->pix_fmt);
    av_assert0(desc);
    if (mjpeg_data->height % (s->frames_per_jpeg << desc->log2_chroma_h)) {
        av_log(avctx, AV_LOG_ERROR, "Invalid height\n");
        return AVERROR_INVALIDDATA;
    }
    /*use the last lot... */
    *data_size = s->mjpeg_data_size;
    avctx->pix_fmt = s->avctx->pix_fmt;
    /* We shouldn't get here if frames_per_jpeg <= 0 because this was rejected
       in init */
    ret = ff_set_dimensions(avctx, mjpeg_data->width, mjpeg_data->height / s->frames_per_jpeg);
    if (ret < 0) {
        av_log(s, AV_LOG_ERROR, "Failed to set dimensions\n");
        return ret;
    }
    if (*data_size) {
        s->picture[1]->extended_data = NULL;
        s->picture[1]->width         = avctx->width;
        s->picture[1]->height        = avctx->height;
        s->picture[1]->format        = avctx->pix_fmt;
        /* ff_init_buffer_info(avctx, &s->picture[1]); */
        smv_img_pnt(s->picture[1]->data, mjpeg_data->data, mjpeg_data->linesize,
                    avctx->pix_fmt, avctx->width, avctx->height, cur_frame);
        for (i = 0; i < AV_NUM_DATA_POINTERS; i++)
            s->picture[1]->linesize[i] = mjpeg_data->linesize[i];
        ret = av_frame_ref(data, s->picture[1]);
    }
    return ret;
}