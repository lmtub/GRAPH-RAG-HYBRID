static int alac_encode_frame(AVCodecContext *avctx, uint8_t *frame,

                             int buf_size, void *data)

{

    AlacEncodeContext *s = avctx->priv_data;

    PutBitContext *pb = &s->pbctx;

    int i, out_bytes, verbatim_flag = 0;



    if (avctx->frame_size > DEFAULT_FRAME_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "input frame size exceeded\n");

        return -1;

    }



    if (buf_size < 2 * s->max_coded_frame_size) {

        av_log(avctx, AV_LOG_ERROR, "buffer size is too small\n");

        return -1;

    }



verbatim:

    init_put_bits(pb, frame, buf_size);



    if (s->compression_level == 0 || verbatim_flag) {

        // Verbatim mode

        const int16_t *samples = data;

        write_frame_header(s, 1);

        for (i = 0; i < avctx->frame_size * avctx->channels; i++) {

            put_sbits(pb, 16, *samples++);

        }

    } else {

        init_sample_buffers(s, data);

        write_frame_header(s, 0);

        write_compressed_frame(s);

    }



    put_bits(pb, 3, 7);

    flush_put_bits(pb);

    out_bytes = put_bits_count(pb) >> 3;



    if (out_bytes > s->max_coded_frame_size) {

        /* frame too large. use verbatim mode */

        if (verbatim_flag || s->compression_level == 0) {

            /* still too large. must be an error. */

            av_log(avctx, AV_LOG_ERROR, "error encoding frame\n");

            return -1;

        }

        verbatim_flag = 1;

        goto verbatim;

    }



    return out_bytes;

}
