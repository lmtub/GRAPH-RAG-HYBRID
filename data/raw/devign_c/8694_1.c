static int get_stats(AVCodecContext *avctx, int eos)

{

#ifdef TH_ENCCTL_2PASS_OUT

    TheoraContext *h = avctx->priv_data;

    uint8_t *buf;

    int bytes;



    bytes = th_encode_ctl(h->t_state, TH_ENCCTL_2PASS_OUT, &buf, sizeof(buf));

    if (bytes < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting first pass stats\n");

        return AVERROR_EXTERNAL;

    }

    if (!eos) {

        void *tmp = av_fast_realloc(h->stats, &h->stats_size,

                                   h->stats_offset + bytes);

        if (!tmp)


        h->stats = tmp;

        memcpy(h->stats + h->stats_offset, buf, bytes);

        h->stats_offset += bytes;

    } else {

        int b64_size = AV_BASE64_SIZE(h->stats_offset);

        // libtheora generates a summary header at the end

        memcpy(h->stats, buf, bytes);

        avctx->stats_out = av_malloc(b64_size);



        av_base64_encode(avctx->stats_out, b64_size, h->stats, h->stats_offset);

    }

    return 0;

#else

    av_log(avctx, AV_LOG_ERROR, "libtheora too old to support 2pass\n");

    return AVERROR(ENOSUP);

#endif

}