static int vmdaudio_decode_frame(AVCodecContext *avctx,

                                 void *data, int *data_size,

                                 AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    VmdAudioContext *s = avctx->priv_data;

    int block_type;

    unsigned char *output_samples = (unsigned char *)data;



    if (buf_size < 16) {

        av_log(avctx, AV_LOG_WARNING, "skipping small junk packet\n");

        *data_size = 0;

        return buf_size;

    }



    block_type = buf[6];

    if (block_type < BLOCK_TYPE_AUDIO || block_type > BLOCK_TYPE_SILENCE) {

        av_log(avctx, AV_LOG_ERROR, "unknown block type: %d\n", block_type);

        return AVERROR(EINVAL);

    }

    buf      += 16;

    buf_size -= 16;



    if (block_type == BLOCK_TYPE_AUDIO) {

        /* the chunk contains audio */

        *data_size = vmdaudio_loadsound(s, output_samples, buf, 0, buf_size);

    } else if (block_type == BLOCK_TYPE_INITIAL) {

        /* initial chunk, may contain audio and silence */

        uint32_t flags = AV_RB32(buf);

        int silent_chunks = av_popcount(flags);

        buf      += 4;

        buf_size -= 4;

        if(*data_size < (s->block_align*silent_chunks + buf_size) * 2)

            return -1;

        *data_size = vmdaudio_loadsound(s, output_samples, buf, silent_chunks, buf_size);

    } else if (block_type == BLOCK_TYPE_SILENCE) {

        /* silent chunk */

        *data_size = vmdaudio_loadsound(s, output_samples, buf, 1, 0);

    }



    return avpkt->size;

}
