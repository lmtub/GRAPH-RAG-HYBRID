int attribute_align_arg avcodec_encode_video2(AVCodecContext *avctx,
                                              AVPacket *avpkt,
                                              const AVFrame *frame,
                                              int *got_packet_ptr)
{
    int ret;
    AVPacket user_pkt = *avpkt;
    int needs_realloc = !user_pkt.data;
    *got_packet_ptr = 0;
    if(CONFIG_FRAME_THREAD_ENCODER &&
       avctx->internal->frame_thread_encoder && (avctx->active_thread_type&FF_THREAD_FRAME))
        return ff_thread_video_encode_frame(avctx, avpkt, frame, got_packet_ptr);
    if ((avctx->flags&CODEC_FLAG_PASS1) && avctx->stats_out)
        avctx->stats_out[0] = '\0';
    if (!(avctx->codec->capabilities & CODEC_CAP_DELAY) && !frame) {
        av_free_packet(avpkt);
        av_init_packet(avpkt);
        avpkt->size = 0;
        return 0;
    }
    if (av_image_check_size(avctx->width, avctx->height, 0, avctx))
        return AVERROR(EINVAL);
    av_assert0(avctx->codec->encode2);
    ret = avctx->codec->encode2(avctx, avpkt, frame, got_packet_ptr);
    av_assert0(ret <= 0);
    if (avpkt->data && avpkt->data == avctx->internal->byte_buffer) {
        needs_realloc = 0;
        if (user_pkt.data) {
            if (user_pkt.size >= avpkt->size) {
                memcpy(user_pkt.data, avpkt->data, avpkt->size);
            } else {
                av_log(avctx, AV_LOG_ERROR, "Provided packet is too small, needs to be %d\n", avpkt->size);
                avpkt->size = user_pkt.size;
                ret = -1;
            }
            avpkt->buf      = user_pkt.buf;
            avpkt->data     = user_pkt.data;
#if FF_API_DESTRUCT_PACKET
FF_DISABLE_DEPRECATION_WARNINGS
            avpkt->destruct = user_pkt.destruct;
FF_ENABLE_DEPRECATION_WARNINGS
#endif
        } else {
            if (av_dup_packet(avpkt) < 0) {
                ret = AVERROR(ENOMEM);
            }
        }
    }
    if (!ret) {
        if (!*got_packet_ptr)
            avpkt->size = 0;
        else if (!(avctx->codec->capabilities & CODEC_CAP_DELAY))
            avpkt->pts = avpkt->dts = frame->pts;
        if (needs_realloc && avpkt->data) {
            ret = av_buffer_realloc(&avpkt->buf, avpkt->size + FF_INPUT_BUFFER_PADDING_SIZE);
            if (ret >= 0)
                avpkt->data = avpkt->buf->data;
        }
        avctx->frame_number++;
    }
    if (ret < 0 || !*got_packet_ptr)
        av_free_packet(avpkt);
    else
        av_packet_merge_side_data(avpkt);
    emms_c();
    return ret;
}