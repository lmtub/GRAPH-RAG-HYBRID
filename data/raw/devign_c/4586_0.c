static int vda_h264_end_frame(AVCodecContext *avctx)

{

    H264Context *h        = avctx->priv_data;

    VDAContext *vda       = avctx->internal->hwaccel_priv_data;

    AVVDAContext *vda_ctx = avctx->hwaccel_context;

    AVFrame *frame        = h->cur_pic_ptr->f;

    uint32_t flush_flags  = 1 << 0; ///< kVDADecoderFlush_emitFrames

    CFDataRef coded_frame;

    OSStatus status;



    if (!vda->bitstream_size)

        return AVERROR_INVALIDDATA;





    coded_frame = CFDataCreate(kCFAllocatorDefault,

                               vda->bitstream,

                               vda->bitstream_size);



    status = VDADecoderDecode(vda_ctx->decoder, 0, coded_frame, NULL);



    if (status == kVDADecoderNoErr)

        status = VDADecoderFlush(vda_ctx->decoder, flush_flags);



    CFRelease(coded_frame);



    if (!vda->frame)

        return AVERROR_UNKNOWN;



    if (status != kVDADecoderNoErr) {

        av_log(avctx, AV_LOG_ERROR, "Failed to decode frame (%d)\n", status);

        return AVERROR_UNKNOWN;

    }



    av_buffer_unref(&frame->buf[0]);



    frame->buf[0] = av_buffer_create((uint8_t*)vda->frame,

                                     sizeof(vda->frame),

                                     release_buffer, NULL,

                                     AV_BUFFER_FLAG_READONLY);

    if (!frame->buf)

        return AVERROR(ENOMEM);



    frame->data[3] = (uint8_t*)vda->frame;

    vda->frame = NULL;



    return 0;

}
