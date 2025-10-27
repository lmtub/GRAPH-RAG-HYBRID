dshow_cycle_formats(AVFormatContext *avctx, enum dshowDeviceType devtype,

                    IPin *pin, int *pformat_set)

{

    struct dshow_ctx *ctx = avctx->priv_data;

    IAMStreamConfig *config = NULL;

    AM_MEDIA_TYPE *type = NULL;

    int format_set = 0;

    void *caps = NULL;

    int i, n, size;



    if (IPin_QueryInterface(pin, &IID_IAMStreamConfig, (void **) &config) != S_OK)

        return;

    if (IAMStreamConfig_GetNumberOfCapabilities(config, &n, &size) != S_OK)

        goto end;



    caps = av_malloc(size);

    if (!caps)

        goto end;



    for (i = 0; i < n && !format_set; i++) {

        IAMStreamConfig_GetStreamCaps(config, i, &type, (void *) caps);



#if DSHOWDEBUG

        ff_print_AM_MEDIA_TYPE(type);

#endif



        if (devtype == VideoDevice) {

            VIDEO_STREAM_CONFIG_CAPS *vcaps = caps;

            BITMAPINFOHEADER *bih;

            int64_t *fr;

            const AVCodecTag *const tags[] = { avformat_get_riff_video_tags(), NULL };

#if DSHOWDEBUG

            ff_print_VIDEO_STREAM_CONFIG_CAPS(vcaps);

#endif

            if (IsEqualGUID(&type->formattype, &FORMAT_VideoInfo)) {

                VIDEOINFOHEADER *v = (void *) type->pbFormat;

                fr = &v->AvgTimePerFrame;

                bih = &v->bmiHeader;

            } else if (IsEqualGUID(&type->formattype, &FORMAT_VideoInfo2)) {

                VIDEOINFOHEADER2 *v = (void *) type->pbFormat;

                fr = &v->AvgTimePerFrame;

                bih = &v->bmiHeader;

            } else {

                goto next;

            }

            if (!pformat_set) {

                enum AVPixelFormat pix_fmt = dshow_pixfmt(bih->biCompression, bih->biBitCount);

                if (pix_fmt == AV_PIX_FMT_NONE) {

                    enum AVCodecID codec_id = av_codec_get_id(tags, bih->biCompression);

                    AVCodec *codec = avcodec_find_decoder(codec_id);

                    if (codec_id == AV_CODEC_ID_NONE || !codec) {

                        av_log(avctx, AV_LOG_INFO, "  unknown compression type 0x%X", (int) bih->biCompression);

                    } else {

                        av_log(avctx, AV_LOG_INFO, "  vcodec=%s", codec->name);

                    }

                } else {

                    av_log(avctx, AV_LOG_INFO, "  pixel_format=%s", av_get_pix_fmt_name(pix_fmt));

                }

                av_log(avctx, AV_LOG_INFO, "  min s=%ldx%ld fps=%g max s=%ldx%ld fps=%g\n",

                       vcaps->MinOutputSize.cx, vcaps->MinOutputSize.cy,

                       1e7 / vcaps->MaxFrameInterval,

                       vcaps->MaxOutputSize.cx, vcaps->MaxOutputSize.cy,

                       1e7 / vcaps->MinFrameInterval);

                continue;

            }

            if (ctx->video_codec_id != AV_CODEC_ID_RAWVIDEO) {

                if (ctx->video_codec_id != av_codec_get_id(tags, bih->biCompression))

                    goto next;

            }

            if (ctx->pixel_format != AV_PIX_FMT_NONE &&

                ctx->pixel_format != dshow_pixfmt(bih->biCompression, bih->biBitCount)) {

                goto next;

            }

            if (ctx->framerate) {

                int64_t framerate = ((int64_t) ctx->requested_framerate.den*10000000)

                                            /  ctx->requested_framerate.num;

                if (framerate > vcaps->MaxFrameInterval ||

                    framerate < vcaps->MinFrameInterval)

                    goto next;

                *fr = framerate;

            }

            if (ctx->requested_width && ctx->requested_height) {

                if (ctx->requested_width  > vcaps->MaxOutputSize.cx ||

                    ctx->requested_width  < vcaps->MinOutputSize.cx ||

                    ctx->requested_height > vcaps->MaxOutputSize.cy ||

                    ctx->requested_height < vcaps->MinOutputSize.cy)

                    goto next;

                bih->biWidth  = ctx->requested_width;

                bih->biHeight = ctx->requested_height;

            }

        } else {

            AUDIO_STREAM_CONFIG_CAPS *acaps = caps;

            WAVEFORMATEX *fx;

#if DSHOWDEBUG

            ff_print_AUDIO_STREAM_CONFIG_CAPS(acaps);

#endif

            if (IsEqualGUID(&type->formattype, &FORMAT_WaveFormatEx)) {

                fx = (void *) type->pbFormat;

            } else {

                goto next;

            }

            if (!pformat_set) {

                av_log(avctx, AV_LOG_INFO, "  min ch=%lu bits=%lu rate=%6lu max ch=%lu bits=%lu rate=%6lu\n",

                       acaps->MinimumChannels, acaps->MinimumBitsPerSample, acaps->MinimumSampleFrequency,

                       acaps->MaximumChannels, acaps->MaximumBitsPerSample, acaps->MaximumSampleFrequency);

                continue;

            }

            if (ctx->sample_rate) {

                if (ctx->sample_rate > acaps->MaximumSampleFrequency ||

                    ctx->sample_rate < acaps->MinimumSampleFrequency)

                    goto next;

                fx->nSamplesPerSec = ctx->sample_rate;

            }

            if (ctx->sample_size) {

                if (ctx->sample_size > acaps->MaximumBitsPerSample ||

                    ctx->sample_size < acaps->MinimumBitsPerSample)

                    goto next;

                fx->wBitsPerSample = ctx->sample_size;

            }

            if (ctx->channels) {

                if (ctx->channels > acaps->MaximumChannels ||

                    ctx->channels < acaps->MinimumChannels)

                    goto next;

                fx->nChannels = ctx->channels;

            }

        }

        if (IAMStreamConfig_SetFormat(config, type) != S_OK)

            goto next;

        format_set = 1;

next:

        if (type->pbFormat)

            CoTaskMemFree(type->pbFormat);

        CoTaskMemFree(type);

    }

end:

    IAMStreamConfig_Release(config);

    if (caps)

        av_free(caps);

    if (pformat_set)

        *pformat_set = format_set;

}
