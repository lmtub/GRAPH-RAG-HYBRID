static int ffmmal_add_packet(AVCodecContext *avctx, AVPacket *avpkt,

                             int is_extradata)

{

    MMALDecodeContext *ctx = avctx->priv_data;

    AVBufferRef *buf = NULL;

    int size = 0;

    uint8_t *data = (uint8_t *)"";

    uint8_t *start;

    int ret = 0;



    if (avpkt->size) {

        if (avpkt->buf) {

            buf = av_buffer_ref(avpkt->buf);

            size = avpkt->size;

            data = avpkt->data;

        } else {

            buf = av_buffer_alloc(avpkt->size);

            if (buf) {

                memcpy(buf->data, avpkt->data, avpkt->size);

                size = buf->size;

                data = buf->data;

            }

        }

        if (!buf) {

            ret = AVERROR(ENOMEM);

            goto done;

        }

        if (!is_extradata)

            ctx->packets_sent++;

    } else {

        if (!ctx->packets_sent) {

            // Short-cut the flush logic to avoid upsetting MMAL.

            ctx->eos_sent = 1;

            ctx->eos_received = 1;

            goto done;

        }

    }



    start = data;



    do {

        FFBufferEntry *buffer = av_mallocz(sizeof(*buffer));

        if (!buffer) {

            ret = AVERROR(ENOMEM);

            goto done;

        }



        buffer->data = data;

        buffer->length = FFMIN(size, ctx->decoder->input[0]->buffer_size);



        if (is_extradata)

            buffer->flags |= MMAL_BUFFER_HEADER_FLAG_CONFIG;



        if (data == start)

            buffer->flags |= MMAL_BUFFER_HEADER_FLAG_FRAME_START;



        data += buffer->length;

        size -= buffer->length;



        buffer->pts = avpkt->pts == AV_NOPTS_VALUE ? MMAL_TIME_UNKNOWN : avpkt->pts;

        buffer->dts = avpkt->dts == AV_NOPTS_VALUE ? MMAL_TIME_UNKNOWN : avpkt->dts;



        if (!size)

            buffer->flags |= MMAL_BUFFER_HEADER_FLAG_FRAME_END;



        if (!buffer->length) {

            buffer->flags |= MMAL_BUFFER_HEADER_FLAG_EOS;

            ctx->eos_sent = 1;

        }



        if (buf) {

            buffer->ref = av_buffer_ref(buf);

            if (!buffer->ref) {

                av_free(buffer);

                ret = AVERROR(ENOMEM);

                goto done;

            }

        }



        // Insert at end of the list

        if (!ctx->waiting_buffers)

            ctx->waiting_buffers = buffer;

        if (ctx->waiting_buffers_tail)

            ctx->waiting_buffers_tail->next = buffer;

        ctx->waiting_buffers_tail = buffer;

    } while (size);



done:

    av_buffer_unref(&buf);

    return ret;

}
