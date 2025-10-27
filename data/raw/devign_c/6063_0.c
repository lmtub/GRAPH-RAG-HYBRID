static int eightsvx_decode_frame(AVCodecContext *avctx, void *data,

                                 int *got_frame_ptr, AVPacket *avpkt)

{

    EightSvxContext *esc = avctx->priv_data;

    int n, out_data_size;

    int ch, ret;

    uint8_t *src;



    /* decode and interleave the first packet */

    if (!esc->samples && avpkt) {

        int packet_size = avpkt->size;



        if (packet_size % avctx->channels) {

            av_log(avctx, AV_LOG_WARNING, "Packet with odd size, ignoring last byte\n");

            if (packet_size < avctx->channels)

                return packet_size;

            packet_size -= packet_size % avctx->channels;

        }

        esc->samples_size = !esc->table ?

            packet_size : avctx->channels + (packet_size-avctx->channels) * 2;

        if (!(esc->samples = av_malloc(esc->samples_size)))

            return AVERROR(ENOMEM);



        /* decompress */

        if (esc->table) {

            const uint8_t *buf = avpkt->data;

            uint8_t *dst;

            int buf_size = avpkt->size;

            int i, n = esc->samples_size;



            if (buf_size < 2) {

                av_log(avctx, AV_LOG_ERROR, "packet size is too small\n");

                return AVERROR(EINVAL);

            }



            /* the uncompressed starting value is contained in the first byte */

            dst = esc->samples;

            for (i = 0; i < avctx->channels; i++) {

                *(dst++) = buf[0]+128;

                delta_decode(dst, buf + 1, buf_size / avctx->channels - 1, (buf[0]+128)&0xFF, esc->table);

                buf += buf_size / avctx->channels;

                dst += n / avctx->channels - 1;

            }

        } else {

            raw_decode(esc->samples, avpkt->data, esc->samples_size);

        }

    }



    /* get output buffer */

    av_assert1(!(esc->samples_size % avctx->channels || esc->samples_idx % avctx->channels));

    esc->frame.nb_samples = FFMIN(MAX_FRAME_SIZE, esc->samples_size - esc->samples_idx)  / avctx->channels;

    if ((ret = avctx->get_buffer(avctx, &esc->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    *got_frame_ptr   = 1;

    *(AVFrame *)data = esc->frame;



    out_data_size = esc->frame.nb_samples;

    for (ch = 0; ch<avctx->channels; ch++) {

        src = esc->samples + esc->samples_idx / avctx->channels + ch * esc->samples_size / avctx->channels;

        memcpy(esc->frame.data[ch], src, out_data_size);

    }

    out_data_size *= avctx->channels;

    esc->samples_idx += out_data_size;



    return esc->table ?

        (avctx->frame_number == 0)*2 + out_data_size / 2 :

        out_data_size;

}
