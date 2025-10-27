static int decode_frame_byterun1(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    IffContext *s = avctx->priv_data;

    const uint8_t *buf = avpkt->data;

    unsigned buf_size = avpkt->size;

    const uint8_t *buf_end = buf+buf_size;

    unsigned y, plane, x;



    if (avctx->reget_buffer(avctx, &s->frame) < 0){

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    for(y = 0; y < avctx->height ; y++ ) {

        uint8_t *row = &s->frame.data[0][ y*s->frame.linesize[0] ];

        if (avctx->codec_tag == MKTAG('I','L','B','M')) { //interleaved

            memset(row, 0, avctx->pix_fmt == PIX_FMT_PAL8 ? avctx->width : (avctx->width * 4));

            for (plane = 0; plane < avctx->bits_per_coded_sample; plane++) {

                for(x = 0; x < s->planesize && buf < buf_end; ) {

                    int8_t value = *buf++;

                    unsigned length;

                    if (value >= 0) {

                        length = value + 1;

                        memcpy(s->planebuf + x, buf, FFMIN3(length, s->planesize - x, buf_end - buf));

                        buf += length;

                    } else if (value > -128) {

                        length = -value + 1;

                        memset(s->planebuf + x, *buf++, FFMIN(length, s->planesize - x));

                    } else { //noop

                        continue;

                    }

                    x += length;

                }

                if (avctx->pix_fmt == PIX_FMT_PAL8) {

                    decodeplane8(row, s->planebuf, s->planesize, avctx->bits_per_coded_sample, plane);

                } else { //PIX_FMT_BGR32

                    decodeplane32((uint32_t *) row, s->planebuf, s->planesize, avctx->bits_per_coded_sample, plane);

                }

            }

        } else {

            for(x = 0; x < avctx->width && buf < buf_end; ) {

                int8_t value = *buf++;

                unsigned length;

                if (value >= 0) {

                    length = value + 1;

                    memcpy(row + x, buf, FFMIN3(length, buf_end - buf, avctx->width - x));

                    buf += length;

                } else if (value > -128) {

                    length = -value + 1;

                    memset(row + x, *buf++, FFMIN(length, avctx->width - x));

                } else { //noop

                    continue;

                }

                x += length;

            }

        }

    }



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;

    return buf_size;

}
