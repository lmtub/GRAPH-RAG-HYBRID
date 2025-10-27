static int ir2_decode_frame(AVCodecContext *avctx,

                        void *data, int *data_size,

                        AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    Ir2Context * const s = avctx->priv_data;

    AVFrame *picture = data;

    AVFrame * const p= (AVFrame*)&s->picture;

    int start;



    if(p->data[0])

        avctx->release_buffer(avctx, p);



    p->reference = 1;

    p->buffer_hints = FF_BUFFER_HINTS_VALID | FF_BUFFER_HINTS_PRESERVE | FF_BUFFER_HINTS_REUSABLE;

    if (avctx->reget_buffer(avctx, p)) {

        av_log(s->avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return -1;

    }



    s->decode_delta = buf[18];



    /* decide whether frame uses deltas or not */

#ifndef ALT_BITSTREAM_READER_LE

    for (i = 0; i < buf_size; i++)

        buf[i] = av_reverse[buf[i]];

#endif

    start = 48; /* hardcoded for now */



    init_get_bits(&s->gb, buf + start, buf_size - start);



    if (s->decode_delta) { /* intraframe */

        ir2_decode_plane(s, avctx->width, avctx->height,

                         s->picture.data[0], s->picture.linesize[0], ir2_luma_table);

        /* swapped U and V */

        ir2_decode_plane(s, avctx->width >> 2, avctx->height >> 2,

                         s->picture.data[2], s->picture.linesize[2], ir2_luma_table);

        ir2_decode_plane(s, avctx->width >> 2, avctx->height >> 2,

                         s->picture.data[1], s->picture.linesize[1], ir2_luma_table);

    } else { /* interframe */

        ir2_decode_plane_inter(s, avctx->width, avctx->height,

                         s->picture.data[0], s->picture.linesize[0], ir2_luma_table);

        /* swapped U and V */

        ir2_decode_plane_inter(s, avctx->width >> 2, avctx->height >> 2,

                         s->picture.data[2], s->picture.linesize[2], ir2_luma_table);

        ir2_decode_plane_inter(s, avctx->width >> 2, avctx->height >> 2,

                         s->picture.data[1], s->picture.linesize[1], ir2_luma_table);

    }



    *picture= *(AVFrame*)&s->picture;

    *data_size = sizeof(AVPicture);



    return buf_size;

}
