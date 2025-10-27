static int alac_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                             const AVFrame *frame, int *got_packet_ptr)

{

    AlacEncodeContext *s = avctx->priv_data;

    int out_bytes, max_frame_size, ret;



    s->frame_size = frame->nb_samples;



    if (frame->nb_samples < DEFAULT_FRAME_SIZE)

        max_frame_size = get_max_frame_size(s->frame_size, avctx->channels,

                                            avctx->bits_per_raw_sample);

    else

        max_frame_size = s->max_coded_frame_size;



    if ((ret = ff_alloc_packet2(avctx, avpkt, 2 * max_frame_size)))

        return ret;



    /* use verbatim mode for compression_level 0 */

    if (s->compression_level) {

        s->verbatim   = 0;

        s->extra_bits = avctx->bits_per_raw_sample - 16;

    } else {

        s->verbatim   = 1;

        s->extra_bits = 0;

    }



    out_bytes = write_frame(s, avpkt, frame->extended_data);



    if (out_bytes > max_frame_size) {

        /* frame too large. use verbatim mode */

        s->verbatim = 1;

        s->extra_bits = 0;

        out_bytes = write_frame(s, avpkt, frame->extended_data);

    }



    avpkt->size = out_bytes;

    *got_packet_ptr = 1;

    return 0;

}
