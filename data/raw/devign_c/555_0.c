static int mp3lame_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                                const AVFrame *frame, int *got_packet_ptr)

{

    LAMEContext *s = avctx->priv_data;

    MPADecodeHeader hdr;

    int len, ret, ch;

    int lame_result;

    uint32_t h;



    if (frame) {

        switch (avctx->sample_fmt) {

        case AV_SAMPLE_FMT_S16P:

            ENCODE_BUFFER(lame_encode_buffer, int16_t, frame->data);

            break;

        case AV_SAMPLE_FMT_S32P:

            ENCODE_BUFFER(lame_encode_buffer_int, int32_t, frame->data);

            break;

        case AV_SAMPLE_FMT_FLTP:

            if (frame->linesize[0] < 4 * FFALIGN(frame->nb_samples, 8)) {

                av_log(avctx, AV_LOG_ERROR, "inadequate AVFrame plane padding\n");

                return AVERROR(EINVAL);

            }

            for (ch = 0; ch < avctx->channels; ch++) {

                s->fdsp.vector_fmul_scalar(s->samples_flt[ch],

                                           (const float *)frame->data[ch],

                                           32768.0f,

                                           FFALIGN(frame->nb_samples, 8));

            }

            ENCODE_BUFFER(lame_encode_buffer_float, float, s->samples_flt);

            break;

        default:

            return AVERROR_BUG;

        }

    } else {

        lame_result = lame_encode_flush(s->gfp, s->buffer + s->buffer_index,

                                        s->buffer_size - s->buffer_index);

    }

    if (lame_result < 0) {

        if (lame_result == -1) {

            av_log(avctx, AV_LOG_ERROR,

                   "lame: output buffer too small (buffer index: %d, free bytes: %d)\n",

                   s->buffer_index, s->buffer_size - s->buffer_index);

        }

        return -1;

    }

    s->buffer_index += lame_result;

    ret = realloc_buffer(s);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "error reallocating output buffer\n");

        return ret;

    }



    /* add current frame to the queue */

    if (frame) {

        if ((ret = ff_af_queue_add(&s->afq, frame)) < 0)

            return ret;

    }



    /* Move 1 frame from the LAME buffer to the output packet, if available.

       We have to parse the first frame header in the output buffer to

       determine the frame size. */

    if (s->buffer_index < 4)

        return 0;

    h = AV_RB32(s->buffer);

    if (ff_mpa_check_header(h) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Invalid mp3 header at start of buffer\n");

        return AVERROR_BUG;

    }

    if (avpriv_mpegaudio_decode_header(&hdr, h)) {

        av_log(avctx, AV_LOG_ERROR, "free format output not supported\n");

        return -1;

    }

    len = hdr.frame_size;

    ff_dlog(avctx, "in:%d packet-len:%d index:%d\n", avctx->frame_size, len,

            s->buffer_index);

    if (len <= s->buffer_index) {

        if ((ret = ff_alloc_packet(avpkt, len))) {

            av_log(avctx, AV_LOG_ERROR, "Error getting output packet\n");

            return ret;

        }

        memcpy(avpkt->data, s->buffer, len);

        s->buffer_index -= len;

        memmove(s->buffer, s->buffer + len, s->buffer_index);



        /* Get the next frame pts/duration */

        ff_af_queue_remove(&s->afq, avctx->frame_size, &avpkt->pts,

                           &avpkt->duration);



        avpkt->size = len;

        *got_packet_ptr = 1;

    }

    return 0;

}
