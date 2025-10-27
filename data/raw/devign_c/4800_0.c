static av_cold int aac_encode_init(AVCodecContext *avctx)

{

    AACContext *s = avctx->priv_data;

    int ret = AVERROR(EINVAL);

    AACENC_InfoStruct info = { 0 };

    CHANNEL_MODE mode;

    AACENC_ERROR err;

    int aot = FF_PROFILE_AAC_LOW + 1;

    int sce = 0, cpe = 0;



    if ((err = aacEncOpen(&s->handle, 0, avctx->channels)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to open the encoder: %s\n",

               aac_get_error(err));

        goto error;

    }



    if (avctx->profile != FF_PROFILE_UNKNOWN)

        aot = avctx->profile + 1;



    if ((err = aacEncoder_SetParam(s->handle, AACENC_AOT, aot)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to set the AOT %d: %s\n",

               aot, aac_get_error(err));

        goto error;

    }



    if (aot == FF_PROFILE_AAC_ELD + 1 && s->eld_sbr) {

        if ((err = aacEncoder_SetParam(s->handle, AACENC_SBR_MODE,

                                       1)) != AACENC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to enable SBR for ELD: %s\n",

                   aac_get_error(err));

            goto error;

        }

    }



    if ((err = aacEncoder_SetParam(s->handle, AACENC_SAMPLERATE,

                                   avctx->sample_rate)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to set the sample rate %d: %s\n",

               avctx->sample_rate, aac_get_error(err));

        goto error;

    }



    switch (avctx->channels) {

    case 1: mode = MODE_1;       sce = 1; cpe = 0; break;

    case 2: mode = MODE_2;       sce = 0; cpe = 1; break;

    case 3: mode = MODE_1_2;     sce = 1; cpe = 1; break;

    case 4: mode = MODE_1_2_1;   sce = 2; cpe = 1; break;

    case 5: mode = MODE_1_2_2;   sce = 1; cpe = 2; break;

    case 6: mode = MODE_1_2_2_1; sce = 2; cpe = 2; break;

    default:

        av_log(avctx, AV_LOG_ERROR,

               "Unsupported number of channels %d\n", avctx->channels);

        goto error;

    }



    if ((err = aacEncoder_SetParam(s->handle, AACENC_CHANNELMODE,

                                   mode)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR,

               "Unable to set channel mode %d: %s\n", mode, aac_get_error(err));

        goto error;

    }



    if ((err = aacEncoder_SetParam(s->handle, AACENC_CHANNELORDER,

                                   1)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR,

               "Unable to set wav channel order %d: %s\n",

               mode, aac_get_error(err));

        goto error;

    }



    if (avctx->flags & CODEC_FLAG_QSCALE || s->vbr) {

        int mode = s->vbr ? s->vbr : avctx->global_quality;

        if (mode <  1 || mode > 5) {

            av_log(avctx, AV_LOG_WARNING,

                   "VBR quality %d out of range, should be 1-5\n", mode);

            mode = av_clip(mode, 1, 5);

        }

        av_log(avctx, AV_LOG_WARNING,

               "Note, the VBR setting is unsupported and only works with "

               "some parameter combinations\n");

        if ((err = aacEncoder_SetParam(s->handle, AACENC_BITRATEMODE,

                                       mode)) != AACENC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set the VBR bitrate mode %d: %s\n",

                   mode, aac_get_error(err));

            goto error;

        }

    } else {

        if (avctx->bit_rate <= 0) {

            if (avctx->profile == FF_PROFILE_AAC_HE_V2) {

                sce = 1;

                cpe = 0;

            }

            avctx->bit_rate = (96*sce + 128*cpe) * avctx->sample_rate / 44;

            if (avctx->profile == FF_PROFILE_AAC_HE ||

                avctx->profile == FF_PROFILE_AAC_HE_V2 ||

                s->eld_sbr)

                avctx->bit_rate /= 2;

        }

        if ((err = aacEncoder_SetParam(s->handle, AACENC_BITRATE,

                                       avctx->bit_rate)) != AACENC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set the bitrate %d: %s\n",

                   avctx->bit_rate, aac_get_error(err));

            goto error;

        }

    }



    /* Choose bitstream format - if global header is requested, use

     * raw access units, otherwise use ADTS. */

    if ((err = aacEncoder_SetParam(s->handle, AACENC_TRANSMUX,

                                   avctx->flags & CODEC_FLAG_GLOBAL_HEADER ? 0 : s->latm ? 10 : 2)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to set the transmux format: %s\n",

               aac_get_error(err));

        goto error;

    }



    if (s->latm && s->header_period) {

        if ((err = aacEncoder_SetParam(s->handle, AACENC_HEADER_PERIOD,

                                       s->header_period)) != AACENC_OK) {

             av_log(avctx, AV_LOG_ERROR, "Unable to set header period: %s\n",

                    aac_get_error(err));

             goto error;

        }

    }



    /* If no signaling mode is chosen, use explicit hierarchical signaling

     * if using mp4 mode (raw access units, with global header) and

     * implicit signaling if using ADTS. */

    if (s->signaling < 0)

        s->signaling = avctx->flags & CODEC_FLAG_GLOBAL_HEADER ? 2 : 0;



    if ((err = aacEncoder_SetParam(s->handle, AACENC_SIGNALING_MODE,

                                   s->signaling)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to set signaling mode %d: %s\n",

               s->signaling, aac_get_error(err));

        goto error;

    }



    if ((err = aacEncoder_SetParam(s->handle, AACENC_AFTERBURNER,

                                   s->afterburner)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to set afterburner to %d: %s\n",

               s->afterburner, aac_get_error(err));

        goto error;

    }



    if (avctx->cutoff > 0) {

        if (avctx->cutoff < (avctx->sample_rate + 255) >> 8) {

            av_log(avctx, AV_LOG_ERROR, "cutoff valid range is %d-20000\n",

                   (avctx->sample_rate + 255) >> 8);

            goto error;

        }

        if ((err = aacEncoder_SetParam(s->handle, AACENC_BANDWIDTH,

                                       avctx->cutoff)) != AACENC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set the encoder bandwidth to %d: %s\n",

                   avctx->cutoff, aac_get_error(err));

            goto error;

        }

    }



    if ((err = aacEncEncode(s->handle, NULL, NULL, NULL, NULL)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to initialize the encoder: %s\n",

               aac_get_error(err));

        return AVERROR(EINVAL);

    }



    if ((err = aacEncInfo(s->handle, &info)) != AACENC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to get encoder info: %s\n",

               aac_get_error(err));

        goto error;

    }



#if FF_API_OLD_ENCODE_AUDIO

    avctx->coded_frame = avcodec_alloc_frame();

    if (!avctx->coded_frame) {

        ret = AVERROR(ENOMEM);

        goto error;

    }

#endif

    avctx->frame_size = info.frameLength;

    avctx->delay      = info.encoderDelay;

    ff_af_queue_init(avctx, &s->afq);



    if (avctx->flags & CODEC_FLAG_GLOBAL_HEADER) {

        avctx->extradata_size = info.confSize;

        avctx->extradata      = av_mallocz(avctx->extradata_size +

                                           FF_INPUT_BUFFER_PADDING_SIZE);

        if (!avctx->extradata) {

            ret = AVERROR(ENOMEM);

            goto error;

        }



        memcpy(avctx->extradata, info.confBuf, info.confSize);

    }

    return 0;

error:

    aac_encode_close(avctx);

    return ret;

}
