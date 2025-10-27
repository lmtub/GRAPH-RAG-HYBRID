static int dfa_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    DfaContext *s = avctx->priv_data;

    const uint8_t *buf = avpkt->data;

    const uint8_t *buf_end = avpkt->data + avpkt->size;

    const uint8_t *tmp_buf;

    uint32_t chunk_type, chunk_size;

    uint8_t *dst;

    int ret;

    int i, pal_elems;



    if (s->pic.data[0])

        avctx->release_buffer(avctx, &s->pic);



    if ((ret = avctx->get_buffer(avctx, &s->pic))) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    while (buf < buf_end) {

        chunk_size = AV_RL32(buf + 4);

        chunk_type = AV_RL32(buf + 8);

        buf += 12;

        if (buf_end - buf < chunk_size) {

            av_log(avctx, AV_LOG_ERROR, "Chunk size is too big (%d bytes)\n", chunk_size);

            return -1;

        }

        if (!chunk_type)

            break;

        if (chunk_type == 1) {

            pal_elems = FFMIN(chunk_size / 3, 256);

            tmp_buf = buf;

            for (i = 0; i < pal_elems; i++) {

                s->pal[i] = bytestream_get_be24(&tmp_buf) << 2;

                s->pal[i] |= (s->pal[i] >> 6) & 0x333;

            }

            s->pic.palette_has_changed = 1;

        } else if (chunk_type <= 9) {

            if (decoder[chunk_type - 2](s->frame_buf, avctx->width, avctx->height,

                                        buf, buf + chunk_size)) {

                av_log(avctx, AV_LOG_ERROR, "Error decoding %s chunk\n",

                       chunk_name[chunk_type - 2]);

                return -1;

            }

        } else {

            av_log(avctx, AV_LOG_WARNING, "Ignoring unknown chunk type %d\n",

                   chunk_type);

        }

        buf += chunk_size;

    }



    buf = s->frame_buf;

    dst = s->pic.data[0];

    for (i = 0; i < avctx->height; i++) {

        memcpy(dst, buf, avctx->width);

        dst += s->pic.linesize[0];

        buf += avctx->width;

    }

    memcpy(s->pic.data[1], s->pal, sizeof(s->pal));



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->pic;



    return avpkt->size;

}
