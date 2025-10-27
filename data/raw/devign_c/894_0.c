static int h264_extradata_to_annexb(AVCodecContext *avctx, const int padding)

{

    uint16_t unit_size;

    uint64_t total_size                 = 0;

    uint8_t *out                        = NULL, unit_nb, sps_done = 0,

             sps_seen                   = 0, pps_seen = 0;

    const uint8_t *extradata            = avctx->extradata + 4;

    static const uint8_t nalu_header[4] = { 0, 0, 0, 1 };

    int length_size = (*extradata++ & 0x3) + 1; // retrieve length coded size



    /* retrieve sps and pps unit(s) */

    unit_nb = *extradata++ & 0x1f; /* number of sps unit(s) */

    if (!unit_nb) {

        goto pps;

    } else {

        sps_seen = 1;

    }



    while (unit_nb--) {

        void *tmp;



        unit_size   = AV_RB16(extradata);

        total_size += unit_size + 4;

        if (total_size > INT_MAX - padding ||

            extradata + 2 + unit_size > avctx->extradata +

            avctx->extradata_size) {

            av_free(out);

            return AVERROR(EINVAL);

        }

        tmp = av_realloc(out, total_size + padding);

        if (!tmp) {

            av_free(out);

            return AVERROR(ENOMEM);

        }

        out = tmp;

        memcpy(out + total_size - unit_size - 4, nalu_header, 4);

        memcpy(out + total_size - unit_size, extradata + 2, unit_size);

        extradata += 2 + unit_size;

pps:

        if (!unit_nb && !sps_done++) {

            unit_nb = *extradata++; /* number of pps unit(s) */

            if (unit_nb)

                pps_seen = 1;

        }

    }



    if (out)

        memset(out + total_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);



    if (!sps_seen)

        av_log(avctx, AV_LOG_WARNING,

               "Warning: SPS NALU missing or invalid. "

               "The resulting stream may not play.\n");



    if (!pps_seen)

        av_log(avctx, AV_LOG_WARNING,

               "Warning: PPS NALU missing or invalid. "

               "The resulting stream may not play.\n");



    av_free(avctx->extradata);

    avctx->extradata      = out;

    avctx->extradata_size = total_size;



    return length_size;

}
