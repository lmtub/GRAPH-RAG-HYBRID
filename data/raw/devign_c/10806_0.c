static int ffmal_update_format(AVCodecContext *avctx)

{

    MMALDecodeContext *ctx = avctx->priv_data;

    MMAL_STATUS_T status;

    int ret = 0;

    MMAL_COMPONENT_T *decoder = ctx->decoder;

    MMAL_ES_FORMAT_T *format_out = decoder->output[0]->format;



    ffmmal_poolref_unref(ctx->pool_out);

    if (!(ctx->pool_out = av_mallocz(sizeof(*ctx->pool_out)))) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    atomic_store(&ctx->pool_out->refcount, 1);



    if (!format_out)

        goto fail;



    if ((status = mmal_port_parameter_set_uint32(decoder->output[0], MMAL_PARAMETER_EXTRA_BUFFERS, ctx->extra_buffers)))

        goto fail;



    if ((status = mmal_port_parameter_set_boolean(decoder->output[0], MMAL_PARAMETER_VIDEO_INTERPOLATE_TIMESTAMPS, 0)))

        goto fail;



    if (avctx->pix_fmt == AV_PIX_FMT_MMAL) {

        format_out->encoding = MMAL_ENCODING_OPAQUE;

    } else {

        format_out->encoding_variant = format_out->encoding = MMAL_ENCODING_I420;

    }



    if ((status = mmal_port_format_commit(decoder->output[0])))

        goto fail;



    if ((ret = ff_set_dimensions(avctx, format_out->es->video.crop.x + format_out->es->video.crop.width,

                                        format_out->es->video.crop.y + format_out->es->video.crop.height)) < 0)

        goto fail;



    if (format_out->es->video.par.num && format_out->es->video.par.den) {

        avctx->sample_aspect_ratio.num = format_out->es->video.par.num;

        avctx->sample_aspect_ratio.den = format_out->es->video.par.den;

    }



    avctx->colorspace = ffmmal_csp_to_av_csp(format_out->es->video.color_space);



    decoder->output[0]->buffer_size =

        FFMAX(decoder->output[0]->buffer_size_min, decoder->output[0]->buffer_size_recommended);

    decoder->output[0]->buffer_num =

        FFMAX(decoder->output[0]->buffer_num_min, decoder->output[0]->buffer_num_recommended) + ctx->extra_buffers;

    ctx->pool_out->pool = mmal_pool_create(decoder->output[0]->buffer_num,

                                           decoder->output[0]->buffer_size);

    if (!ctx->pool_out->pool) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    return 0;



fail:

    return ret < 0 ? ret : AVERROR_UNKNOWN;

}
