static int filter_packet(void *log_ctx, AVPacket *pkt,

                         AVFormatContext *fmt_ctx, AVBitStreamFilterContext *bsf_ctx)

{

    AVCodecContext *enc_ctx = fmt_ctx->streams[pkt->stream_index]->codec;

    int ret = 0;



    while (bsf_ctx) {

        AVPacket new_pkt = *pkt;

        ret = av_bitstream_filter_filter(bsf_ctx, enc_ctx, NULL,

                                             &new_pkt.data, &new_pkt.size,

                                             pkt->data, pkt->size,

                                             pkt->flags & AV_PKT_FLAG_KEY);

        if (ret == 0 && new_pkt.data != pkt->data && new_pkt.destruct) {

            if ((ret = av_copy_packet(&new_pkt, pkt)) < 0)

                break;

            ret = 1;

        }



        if (ret > 0) {

            av_free_packet(pkt);

            new_pkt.buf = av_buffer_create(new_pkt.data, new_pkt.size,

                                           av_buffer_default_free, NULL, 0);

            if (!new_pkt.buf)

                break;

        }

        if (ret < 0) {

            av_log(log_ctx, AV_LOG_ERROR,

                "Failed to filter bitstream with filter %s for stream %d in file '%s' with codec %s\n",

                bsf_ctx->filter->name, pkt->stream_index, fmt_ctx->filename,

                avcodec_get_name(enc_ctx->codec_id));

        }

        *pkt = new_pkt;



        bsf_ctx = bsf_ctx->next;

    }



    return ret;

}
