static av_cold int svc_encode_init(AVCodecContext *avctx)

{

    SVCContext *s = avctx->priv_data;

    SEncParamExt param = { 0 };

    int err = AVERROR_UNKNOWN;

    int log_level;

    WelsTraceCallback callback_function;

    AVCPBProperties *props;



    // Mingw GCC < 4.7 on x86_32 uses an incorrect/buggy ABI for the WelsGetCodecVersion

    // function (for functions returning larger structs), thus skip the check in those

    // configurations.

#if !defined(_WIN32) || !defined(__GNUC__) || !ARCH_X86_32 || AV_GCC_VERSION_AT_LEAST(4, 7)

    OpenH264Version libver = WelsGetCodecVersion();

    if (memcmp(&libver, &g_stCodecVersion, sizeof(libver))) {

        av_log(avctx, AV_LOG_ERROR, "Incorrect library version loaded\n");

        return AVERROR(EINVAL);

    }

#endif



    if (WelsCreateSVCEncoder(&s->encoder)) {

        av_log(avctx, AV_LOG_ERROR, "Unable to create encoder\n");

        return AVERROR_UNKNOWN;

    }



    // Pass all libopenh264 messages to our callback, to allow ourselves to filter them.

    log_level = WELS_LOG_DETAIL;

    (*s->encoder)->SetOption(s->encoder, ENCODER_OPTION_TRACE_LEVEL, &log_level);



    // Set the logging callback function to one that uses av_log() (see implementation above).

    callback_function = (WelsTraceCallback) libopenh264_trace_callback;

    (*s->encoder)->SetOption(s->encoder, ENCODER_OPTION_TRACE_CALLBACK, (void *)&callback_function);



    // Set the AVCodecContext as the libopenh264 callback context so that it can be passed to av_log().

    (*s->encoder)->SetOption(s->encoder, ENCODER_OPTION_TRACE_CALLBACK_CONTEXT, (void *)&avctx);



    (*s->encoder)->GetDefaultParams(s->encoder, &param);



    param.fMaxFrameRate              = avctx->time_base.den / avctx->time_base.num;

    param.iPicWidth                  = avctx->width;

    param.iPicHeight                 = avctx->height;

    param.iTargetBitrate             = avctx->bit_rate;

    param.iMaxBitrate                = FFMAX(avctx->rc_max_rate, avctx->bit_rate);

    param.iRCMode                    = RC_QUALITY_MODE;

    param.iTemporalLayerNum          = 1;

    param.iSpatialLayerNum           = 1;

    param.bEnableDenoise             = 0;

    param.bEnableBackgroundDetection = 1;

    param.bEnableAdaptiveQuant       = 1;

    param.bEnableFrameSkip           = s->skip_frames;

    param.bEnableLongTermReference   = 0;

    param.iLtrMarkPeriod             = 30;

    param.uiIntraPeriod              = avctx->gop_size;

#if OPENH264_VER_AT_LEAST(1, 4)

    param.eSpsPpsIdStrategy          = CONSTANT_ID;

#else

    param.bEnableSpsPpsIdAddition    = 0;

#endif

    param.bPrefixNalAddingCtrl       = 0;

    param.iLoopFilterDisableIdc      = !s->loopfilter;

    param.iEntropyCodingModeFlag     = 0;

    param.iMultipleThreadIdc         = avctx->thread_count;

    if (s->profile && !strcmp(s->profile, "main"))

        param.iEntropyCodingModeFlag = 1;

    else if (!s->profile && avctx->coder_type == FF_CODER_TYPE_AC)

        param.iEntropyCodingModeFlag = 1;



    param.sSpatialLayers[0].iVideoWidth         = param.iPicWidth;

    param.sSpatialLayers[0].iVideoHeight        = param.iPicHeight;

    param.sSpatialLayers[0].fFrameRate          = param.fMaxFrameRate;

    param.sSpatialLayers[0].iSpatialBitrate     = param.iTargetBitrate;

    param.sSpatialLayers[0].iMaxSpatialBitrate  = param.iMaxBitrate;



    if ((avctx->slices > 1) && (s->max_nal_size)){

        av_log(avctx,AV_LOG_ERROR,"Invalid combination -slices %d and -max_nal_size %d.\n",avctx->slices,s->max_nal_size);

        goto fail;

    }



    if (avctx->slices > 1)

        s->slice_mode = SM_FIXEDSLCNUM_SLICE;



    if (s->max_nal_size)

        s->slice_mode = SM_DYN_SLICE;



    param.sSpatialLayers[0].sSliceCfg.uiSliceMode               = s->slice_mode;

    param.sSpatialLayers[0].sSliceCfg.sSliceArgument.uiSliceNum = avctx->slices;



    if (s->slice_mode == SM_DYN_SLICE) {

        if (s->max_nal_size){

            param.uiMaxNalSize = s->max_nal_size;

            param.sSpatialLayers[0].sSliceCfg.sSliceArgument.uiSliceSizeConstraint = s->max_nal_size;

        } else {

            if (avctx->rtp_payload_size) {

                av_log(avctx,AV_LOG_DEBUG,"Using RTP Payload size for uiMaxNalSize");

                param.uiMaxNalSize = avctx->rtp_payload_size;

                param.sSpatialLayers[0].sSliceCfg.sSliceArgument.uiSliceSizeConstraint = avctx->rtp_payload_size;

            } else {

                av_log(avctx,AV_LOG_ERROR,"Invalid -max_nal_size, specify a valid max_nal_size to use -slice_mode dyn\n");

                goto fail;

            }

        }

    }



    if ((*s->encoder)->InitializeExt(s->encoder, &param) != cmResultSuccess) {

        av_log(avctx, AV_LOG_ERROR, "Initialize failed\n");

        goto fail;

    }



    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

        SFrameBSInfo fbi = { 0 };

        int i, size = 0;

        (*s->encoder)->EncodeParameterSets(s->encoder, &fbi);

        for (i = 0; i < fbi.sLayerInfo[0].iNalCount; i++)

            size += fbi.sLayerInfo[0].pNalLengthInByte[i];

        avctx->extradata = av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE);

        if (!avctx->extradata) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

        avctx->extradata_size = size;

        memcpy(avctx->extradata, fbi.sLayerInfo[0].pBsBuf, size);

    }



    props = ff_add_cpb_side_data(avctx);

    if (!props) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    props->max_bitrate = param.iMaxBitrate;

    props->avg_bitrate = param.iTargetBitrate;



    return 0;



fail:

    svc_encode_close(avctx);

    return err;

}
