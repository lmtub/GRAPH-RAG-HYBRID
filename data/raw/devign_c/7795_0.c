static int h264_mp4toannexb_filter(AVBitStreamFilterContext *bsfc,

                                   AVCodecContext *avctx, const char *args,

                                   uint8_t **poutbuf, int *poutbuf_size,

                                   const uint8_t *buf, int buf_size,

                                   int keyframe)

{

    H264BSFContext *ctx = bsfc->priv_data;

    int i;

    uint8_t unit_type;

    int32_t nal_size;

    uint32_t cumul_size    = 0;

    const uint8_t *buf_end = buf + buf_size;

    int ret = 0;



    /* nothing to filter */

    if (!avctx->extradata || avctx->extradata_size < 6) {

        *poutbuf      = (uint8_t *)buf;

        *poutbuf_size = buf_size;

        return 0;

    }



    /* retrieve sps and pps NAL units from extradata */

    if (!ctx->extradata_parsed) {

        if (args && strstr(args, "private_spspps_buf"))

            ctx->private_spspps = 1;



        ret = h264_extradata_to_annexb(ctx, avctx, AV_INPUT_BUFFER_PADDING_SIZE);

        if (ret < 0)

            return ret;

        ctx->length_size      = ret;

        ctx->new_idr          = 1;

        ctx->idr_sps_seen     = 0;

        ctx->idr_pps_seen     = 0;

        ctx->extradata_parsed = 1;

    }



    *poutbuf_size = 0;

    *poutbuf      = NULL;

    do {

        ret= AVERROR(EINVAL);

        if (buf + ctx->length_size > buf_end)

            goto fail;



        for (nal_size = 0, i = 0; i<ctx->length_size; i++)

            nal_size = (nal_size << 8) | buf[i];



        buf      += ctx->length_size;

        unit_type = *buf & 0x1f;



        if (buf + nal_size > buf_end || nal_size < 0)

            goto fail;



        if (unit_type == 7)

            ctx->idr_sps_seen = ctx->new_idr = 1;

        else if (unit_type == 8) {

            ctx->idr_pps_seen = ctx->new_idr = 1;

            /* if SPS has not been seen yet, prepend the AVCC one to PPS */

            if (!ctx->idr_sps_seen) {

                if (ctx->sps_offset == -1)

                    av_log(avctx, AV_LOG_WARNING, "SPS not present in the stream, nor in AVCC, stream may be unreadable\n");

                else {

                    if ((ret = alloc_and_copy(poutbuf, poutbuf_size,

                                         ctx->spspps_buf + ctx->sps_offset,

                                         ctx->pps_offset != -1 ? ctx->pps_offset : ctx->spspps_size - ctx->sps_offset,

                                         buf, nal_size)) < 0)

                        goto fail;

                    ctx->idr_sps_seen = 1;

                    goto next_nal;

                }

            }

        }



        /* if this is a new IDR picture following an IDR picture, reset the idr flag.

         * Just check first_mb_in_slice to be 0 as this is the simplest solution.

         * This could be checking idr_pic_id instead, but would complexify the parsing. */

        if (!ctx->new_idr && unit_type == 5 && (buf[1] & 0x80))

            ctx->new_idr = 1;



        /* prepend only to the first type 5 NAL unit of an IDR picture, if no sps/pps are already present */

        if (ctx->new_idr && unit_type == 5 && !ctx->idr_sps_seen && !ctx->idr_pps_seen) {

            if ((ret=alloc_and_copy(poutbuf, poutbuf_size,

                               ctx->spspps_buf, ctx->spspps_size,

                               buf, nal_size)) < 0)

                goto fail;

            ctx->new_idr = 0;

        /* if only SPS has been seen, also insert PPS */

        } else if (ctx->new_idr && unit_type == 5 && ctx->idr_sps_seen && !ctx->idr_pps_seen) {

            if (ctx->pps_offset == -1) {

                av_log(avctx, AV_LOG_WARNING, "PPS not present in the stream, nor in AVCC, stream may be unreadable\n");

                if ((ret = alloc_and_copy(poutbuf, poutbuf_size,

                                     NULL, 0, buf, nal_size)) < 0)

                    goto fail;

            } else if ((ret = alloc_and_copy(poutbuf, poutbuf_size,

                                        ctx->spspps_buf + ctx->pps_offset, ctx->spspps_size - ctx->pps_offset,

                                        buf, nal_size)) < 0)

                goto fail;

        } else {

            if ((ret=alloc_and_copy(poutbuf, poutbuf_size,

                               NULL, 0, buf, nal_size)) < 0)

                goto fail;

            if (!ctx->new_idr && unit_type == 1) {

                ctx->new_idr = 1;

                ctx->idr_sps_seen = 0;

                ctx->idr_pps_seen = 0;

            }

        }



next_nal:

        buf        += nal_size;

        cumul_size += nal_size + ctx->length_size;

    } while (cumul_size < buf_size);



    return 1;



fail:

    av_freep(poutbuf);

    *poutbuf_size = 0;

    return ret;

}
