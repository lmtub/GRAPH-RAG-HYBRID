int ff_vda_create_decoder(struct vda_context *vda_ctx,

                          uint8_t *extradata,

                          int extradata_size)

{

    OSStatus status = kVDADecoderNoErr;

    CFNumberRef height;

    CFNumberRef width;

    CFNumberRef format;

    CFDataRef avc_data;

    CFMutableDictionaryRef config_info;

    CFMutableDictionaryRef buffer_attributes;

    CFMutableDictionaryRef io_surface_properties;

    CFNumberRef cv_pix_fmt;



    /* Each VCL NAL in the bistream sent to the decoder

     * is preceded by a 4 bytes length header.

     * Change the avcC atom header if needed, to signal headers of 4 bytes. */

    if (extradata_size >= 4 && (extradata[4] & 0x03) != 0x03) {

        uint8_t *rw_extradata;



        if (!(rw_extradata = av_malloc(extradata_size)))

            return AVERROR(ENOMEM);



        memcpy(rw_extradata, extradata, extradata_size);



        rw_extradata[4] |= 0x03;



        avc_data = CFDataCreate(kCFAllocatorDefault, rw_extradata, extradata_size);



        av_freep(&rw_extradata);

    } else {

        avc_data = CFDataCreate(kCFAllocatorDefault, extradata, extradata_size);

    }



    config_info = CFDictionaryCreateMutable(kCFAllocatorDefault,

                                            4,

                                            &kCFTypeDictionaryKeyCallBacks,

                                            &kCFTypeDictionaryValueCallBacks);



    height   = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &vda_ctx->height);

    width    = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &vda_ctx->width);

    format   = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &vda_ctx->format);



    CFDictionarySetValue(config_info, kVDADecoderConfiguration_Height, height);

    CFDictionarySetValue(config_info, kVDADecoderConfiguration_Width, width);

    CFDictionarySetValue(config_info, kVDADecoderConfiguration_SourceFormat, format);

    CFDictionarySetValue(config_info, kVDADecoderConfiguration_avcCData, avc_data);



    buffer_attributes = CFDictionaryCreateMutable(kCFAllocatorDefault,

                                                  2,

                                                  &kCFTypeDictionaryKeyCallBacks,

                                                  &kCFTypeDictionaryValueCallBacks);

    io_surface_properties = CFDictionaryCreateMutable(kCFAllocatorDefault,

                                                      0,

                                                      &kCFTypeDictionaryKeyCallBacks,

                                                      &kCFTypeDictionaryValueCallBacks);

    cv_pix_fmt      = CFNumberCreate(kCFAllocatorDefault,

                                     kCFNumberSInt32Type,

                                     &vda_ctx->cv_pix_fmt_type);

    CFDictionarySetValue(buffer_attributes,

                         kCVPixelBufferPixelFormatTypeKey,

                         cv_pix_fmt);

    CFDictionarySetValue(buffer_attributes,

                         kCVPixelBufferIOSurfacePropertiesKey,

                         io_surface_properties);



    status = VDADecoderCreate(config_info,

                              buffer_attributes,

                              vda_decoder_callback,

                              vda_ctx,

                              &vda_ctx->decoder);



    CFRelease(height);

    CFRelease(width);

    CFRelease(format);

    CFRelease(avc_data);

    CFRelease(config_info);

    CFRelease(io_surface_properties);

    CFRelease(cv_pix_fmt);

    CFRelease(buffer_attributes);



    return status;

}
