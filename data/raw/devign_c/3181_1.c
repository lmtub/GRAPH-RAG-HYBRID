static int vmdaudio_decode_frame(AVCodecContext *avctx, void *data,

                                 int *got_frame_ptr, AVPacket *avpkt)

{

    AVFrame *frame     = data;

    const uint8_t *buf = avpkt->data;

    const uint8_t *buf_end;

    int buf_size = avpkt->size;

    VmdAudioContext *s = avctx->priv_data;

    int block_type, silent_chunks, audio_chunks;

    int ret;

    uint8_t *output_samples_u8;

    int16_t *output_samples_s16;



    if (buf_size < 16) {

        av_log(avctx, AV_LOG_WARNING, "skipping small junk packet\n");

        *got_frame_ptr = 0;

        return buf_size;

    }



    block_type = buf[6];

    if (block_type < BLOCK_TYPE_AUDIO || block_type > BLOCK_TYPE_SILENCE) {

        av_log(avctx, AV_LOG_ERROR, "unknown block type: %d\n", block_type);

        return AVERROR(EINVAL);

    }

    buf      += 16;

    buf_size -= 16;



    /* get number of silent chunks */

    silent_chunks = 0;

    if (block_type == BLOCK_TYPE_INITIAL) {

        uint32_t flags;

        if (buf_size < 4) {

            av_log(avctx, AV_LOG_ERROR, "packet is too small\n");

            return AVERROR(EINVAL);

        }

        flags         = AV_RB32(buf);

        silent_chunks = av_popcount(flags);

        buf      += 4;

        buf_size -= 4;

    } else if (block_type == BLOCK_TYPE_SILENCE) {

        silent_chunks = 1;

        buf_size = 0; // should already be zero but set it just to be sure

    }



    /* ensure output buffer is large enough */

    audio_chunks = buf_size / s->chunk_size;



    /* get output buffer */

    frame->nb_samples = ((silent_chunks + audio_chunks) * avctx->block_align) /

                        avctx->channels;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }

    output_samples_u8  =            frame->data[0];

    output_samples_s16 = (int16_t *)frame->data[0];



    /* decode silent chunks */

    if (silent_chunks > 0) {

        int silent_size = avctx->block_align * silent_chunks;

        if (s->out_bps == 2) {

            memset(output_samples_s16, 0x00, silent_size * 2);

            output_samples_s16 += silent_size;

        } else {

            memset(output_samples_u8,  0x80, silent_size);

            output_samples_u8 += silent_size;

        }

    }



    /* decode audio chunks */

    if (audio_chunks > 0) {

        buf_end = buf + buf_size;

        while (buf < buf_end) {

            if (s->out_bps == 2) {

                decode_audio_s16(output_samples_s16, buf, s->chunk_size,

                                 avctx->channels);

                output_samples_s16 += avctx->block_align;

            } else {

                memcpy(output_samples_u8, buf, s->chunk_size);

                output_samples_u8  += avctx->block_align;

            }

            buf += s->chunk_size;

        }

    }



    *got_frame_ptr = 1;



    return avpkt->size;

}
