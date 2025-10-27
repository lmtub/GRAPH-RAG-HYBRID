static int jpeg2000_decode_frame(AVCodecContext *avctx, void *data,

                                 int *got_frame, AVPacket *avpkt)

{

    Jpeg2000DecoderContext *s = avctx->priv_data;

    ThreadFrame frame = { .f = data };

    AVFrame *picture = data;

    int tileno, ret;



    s->avctx     = avctx;

    s->buf       = s->buf_start = avpkt->data;

    s->buf_end   = s->buf_start + avpkt->size;

    s->curtileno = 0; // TODO: only one tile in DCI JP2K. to implement for more tiles



    // reduction factor, i.e number of resolution levels to skip

    s->reduction_factor = s->lowres;



    if (s->buf_end - s->buf < 2)

        return AVERROR_INVALIDDATA;



    // check if the image is in jp2 format

    if ((AV_RB32(s->buf) == 12) &&

        (AV_RB32(s->buf + 4) == JP2_SIG_TYPE) &&

        (AV_RB32(s->buf + 8) == JP2_SIG_VALUE)) {

        if (!jp2_find_codestream(s)) {

            av_log(avctx, AV_LOG_ERROR,

                   "Could not find Jpeg2000 codestream atom.\n");

            return AVERROR_INVALIDDATA;

        }

    }



    if (bytestream_get_be16(&s->buf) != JPEG2000_SOC) {

        av_log(avctx, AV_LOG_ERROR, "SOC marker not present\n");

        return AVERROR_INVALIDDATA;

    }

    if (ret = jpeg2000_read_main_headers(s))

        goto end;



    /* get picture buffer */

    if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "ff_thread_get_buffer() failed.\n");

        goto end;

    }

    picture->pict_type = AV_PICTURE_TYPE_I;

    picture->key_frame = 1;



    if (ret = jpeg2000_read_bitstream_packets(s))

        goto end;

    for (tileno = 0; tileno < s->numXtiles * s->numYtiles; tileno++)

        if (ret = jpeg2000_decode_tile(s, s->tile + tileno, picture))

            goto end;



    *got_frame = 1;



end:

    jpeg2000_dec_cleanup(s);

    return ret ? ret : s->buf - s->buf_start;

}
