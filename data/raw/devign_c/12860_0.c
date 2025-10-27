static int vtenc_create_encoder(AVCodecContext   *avctx,

                                CMVideoCodecType codec_type,

                                CFStringRef      profile_level,

                                CFNumberRef      gamma_level,

                                CFDictionaryRef  enc_info,

                                CFDictionaryRef  pixel_buffer_info,

                                VTCompressionSessionRef *session)

{

    VTEncContext *vtctx = avctx->priv_data;

    SInt32       bit_rate = avctx->bit_rate;

    CFNumberRef  bit_rate_num;



    int status = VTCompressionSessionCreate(kCFAllocatorDefault,

                                            avctx->width,

                                            avctx->height,

                                            codec_type,

                                            enc_info,

                                            pixel_buffer_info,

                                            kCFAllocatorDefault,

                                            vtenc_output_callback,

                                            avctx,

                                            session);



    if (status || !vtctx->session) {

        av_log(avctx, AV_LOG_ERROR, "Error: cannot create compression session: %d\n", status);



#if !TARGET_OS_IPHONE

        if (!vtctx->allow_sw) {

            av_log(avctx, AV_LOG_ERROR, "Try -allow_sw 1. The hardware encoder may be busy, or not supported.\n");

        }

#endif



        return AVERROR_EXTERNAL;

    }



    bit_rate_num = CFNumberCreate(kCFAllocatorDefault,

                                  kCFNumberSInt32Type,

                                  &bit_rate);

    if (!bit_rate_num) return AVERROR(ENOMEM);



    status = VTSessionSetProperty(vtctx->session,

                                  kVTCompressionPropertyKey_AverageBitRate,

                                  bit_rate_num);

    CFRelease(bit_rate_num);



    if (status) {

        av_log(avctx, AV_LOG_ERROR, "Error setting bitrate property: %d\n", status);

        return AVERROR_EXTERNAL;

    }



    if (profile_level) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_ProfileLevel,

                                      profile_level);

        if (status) {

            av_log(avctx, AV_LOG_ERROR, "Error setting profile/level property: %d\n", status);

            return AVERROR_EXTERNAL;

        }

    }



    if (avctx->gop_size > 0) {

        CFNumberRef interval = CFNumberCreate(kCFAllocatorDefault,

                                              kCFNumberIntType,

                                              &avctx->gop_size);

        if (!interval) {

            return AVERROR(ENOMEM);

        }



        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_MaxKeyFrameInterval,

                                      interval);

        CFRelease(interval);



        if (status) {

            av_log(avctx, AV_LOG_ERROR, "Error setting 'max key-frame interval' property: %d\n", status);

            return AVERROR_EXTERNAL;

        }

    }



    if (vtctx->frames_before) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_MoreFramesBeforeStart,

                                      kCFBooleanTrue);



        if (status == kVTPropertyNotSupportedErr) {

            av_log(avctx, AV_LOG_WARNING, "frames_before property is not supported on this device. Ignoring.\n");

        } else if (status) {

            av_log(avctx, AV_LOG_ERROR, "Error setting frames_before property: %d\n", status);

        }

    }



    if (vtctx->frames_after) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_MoreFramesAfterEnd,

                                      kCFBooleanTrue);



        if (status == kVTPropertyNotSupportedErr) {

            av_log(avctx, AV_LOG_WARNING, "frames_after property is not supported on this device. Ignoring.\n");

        } else if (status) {

            av_log(avctx, AV_LOG_ERROR, "Error setting frames_after property: %d\n", status);

        }

    }



    if (avctx->sample_aspect_ratio.num != 0) {

        CFNumberRef num;

        CFNumberRef den;

        CFMutableDictionaryRef par;

        AVRational *avpar = &avctx->sample_aspect_ratio;



        av_reduce(&avpar->num, &avpar->den,

                   avpar->num,  avpar->den,

                  0xFFFFFFFF);



        num = CFNumberCreate(kCFAllocatorDefault,

                             kCFNumberIntType,

                             &avpar->num);



        den = CFNumberCreate(kCFAllocatorDefault,

                             kCFNumberIntType,

                             &avpar->den);







        par = CFDictionaryCreateMutable(kCFAllocatorDefault,

                                        2,

                                        &kCFCopyStringDictionaryKeyCallBacks,

                                        &kCFTypeDictionaryValueCallBacks);



        if (!par || !num || !den) {

            if (par) CFRelease(par);

            if (num) CFRelease(num);

            if (den) CFRelease(den);



            return AVERROR(ENOMEM);

        }



        CFDictionarySetValue(

            par,

            kCMFormatDescriptionKey_PixelAspectRatioHorizontalSpacing,

            num);



        CFDictionarySetValue(

            par,

            kCMFormatDescriptionKey_PixelAspectRatioVerticalSpacing,

            den);



        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_PixelAspectRatio,

                                      par);



        CFRelease(par);

        CFRelease(num);

        CFRelease(den);



        if (status) {

            av_log(avctx,

                   AV_LOG_ERROR,

                   "Error setting pixel aspect ratio to %d:%d: %d.\n",

                   avctx->sample_aspect_ratio.num,

                   avctx->sample_aspect_ratio.den,

                   status);



            return AVERROR_EXTERNAL;

        }

    }





    if (vtctx->transfer_function) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_TransferFunction,

                                      vtctx->transfer_function);



        if (status) {

            av_log(avctx, AV_LOG_WARNING, "Could not set transfer function: %d\n", status);

        }

    }





    if (vtctx->ycbcr_matrix) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_YCbCrMatrix,

                                      vtctx->ycbcr_matrix);



        if (status) {

            av_log(avctx, AV_LOG_WARNING, "Could not set ycbcr matrix: %d\n", status);

        }

    }





    if (vtctx->color_primaries) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_ColorPrimaries,

                                      vtctx->color_primaries);



        if (status) {

            av_log(avctx, AV_LOG_WARNING, "Could not set color primaries: %d\n", status);

        }

    }



    if (gamma_level) {

        status = VTSessionSetProperty(vtctx->session,

                                      kCVImageBufferGammaLevelKey,

                                      gamma_level);



        if (status) {

            av_log(avctx, AV_LOG_WARNING, "Could not set gamma level: %d\n", status);

        }

    }



    if (!vtctx->has_b_frames) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_AllowFrameReordering,

                                      kCFBooleanFalse);



        if (status) {

            av_log(avctx, AV_LOG_ERROR, "Error setting 'allow frame reordering' property: %d\n", status);

            return AVERROR_EXTERNAL;

        }

    }



    if (vtctx->entropy != VT_ENTROPY_NOT_SET) {

        CFStringRef entropy = vtctx->entropy == VT_CABAC ?

                                kVTH264EntropyMode_CABAC:

                                kVTH264EntropyMode_CAVLC;



        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_H264EntropyMode,

                                      entropy);



        if (status) {

            av_log(avctx, AV_LOG_ERROR, "Error setting entropy property: %d\n", status);

            return AVERROR_EXTERNAL;

        }

    }



    if (vtctx->realtime) {

        status = VTSessionSetProperty(vtctx->session,

                                      kVTCompressionPropertyKey_RealTime,

                                      kCFBooleanTrue);



        if (status) {

            av_log(avctx, AV_LOG_ERROR, "Error setting realtime property: %d\n", status);

        }

    }



    status = VTCompressionSessionPrepareToEncodeFrames(vtctx->session);

    if (status) {

        av_log(avctx, AV_LOG_ERROR, "Error: cannot prepare encoder: %d\n", status);

        return AVERROR_EXTERNAL;

    }



    return 0;

}
