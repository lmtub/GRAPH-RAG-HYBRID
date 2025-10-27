static int cuvid_decode_frame(AVCodecContext *avctx, void *data, int *got_frame, AVPacket *avpkt)

{

    CuvidContext *ctx = avctx->priv_data;

    AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)ctx->hwdevice->data;

    AVCUDADeviceContext *device_hwctx = device_ctx->hwctx;

    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;

    AVFrame *frame = data;

    CUVIDSOURCEDATAPACKET cupkt;

    AVPacket filter_packet = { 0 };

    AVPacket filtered_packet = { 0 };

    CUdeviceptr mapped_frame = 0;

    int ret = 0, eret = 0;



    if (ctx->bsf && avpkt->size) {

        if ((ret = av_packet_ref(&filter_packet, avpkt)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "av_packet_ref failed\n");

            return ret;

        }



        if ((ret = av_bsf_send_packet(ctx->bsf, &filter_packet)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "av_bsf_send_packet failed\n");

            av_packet_unref(&filter_packet);

            return ret;

        }



        if ((ret = av_bsf_receive_packet(ctx->bsf, &filtered_packet)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "av_bsf_receive_packet failed\n");

            return ret;

        }



        avpkt = &filtered_packet;

    }



    ret = CHECK_CU(cuCtxPushCurrent(cuda_ctx));

    if (ret < 0) {

        av_packet_unref(&filtered_packet);

        return ret;

    }



    memset(&cupkt, 0, sizeof(cupkt));



    if (avpkt->size) {

        cupkt.payload_size = avpkt->size;

        cupkt.payload = avpkt->data;



        if (avpkt->pts != AV_NOPTS_VALUE) {

            cupkt.flags = CUVID_PKT_TIMESTAMP;

            if (avctx->pkt_timebase.num && avctx->pkt_timebase.den)

                cupkt.timestamp = av_rescale_q(avpkt->pts, avctx->pkt_timebase, (AVRational){1, 10000000});

            else

                cupkt.timestamp = avpkt->pts;

        }

    } else {

        cupkt.flags = CUVID_PKT_ENDOFSTREAM;

    }



    ret = CHECK_CU(cuvidParseVideoData(ctx->cuparser, &cupkt));



    av_packet_unref(&filtered_packet);



    if (ret < 0) {

        if (ctx->internal_error)

            ret = ctx->internal_error;

        goto error;

    }



    if (av_fifo_size(ctx->frame_queue)) {

        CUVIDPARSERDISPINFO dispinfo;

        CUVIDPROCPARAMS params;

        unsigned int pitch = 0;

        int offset = 0;

        int i;



        av_fifo_generic_read(ctx->frame_queue, &dispinfo, sizeof(CUVIDPARSERDISPINFO), NULL);



        memset(&params, 0, sizeof(params));

        params.progressive_frame = dispinfo.progressive_frame;

        params.second_field = 0;

        params.top_field_first = dispinfo.top_field_first;



        ret = CHECK_CU(cuvidMapVideoFrame(ctx->cudecoder, dispinfo.picture_index, &mapped_frame, &pitch, &params));

        if (ret < 0)

            goto error;



        if (avctx->pix_fmt == AV_PIX_FMT_CUDA) {

            ret = av_hwframe_get_buffer(ctx->hwframe, frame, 0);

            if (ret < 0) {

                av_log(avctx, AV_LOG_ERROR, "av_hwframe_get_buffer failed\n");

                goto error;

            }



            ret = ff_decode_frame_props(avctx, frame);

            if (ret < 0) {

                av_log(avctx, AV_LOG_ERROR, "ff_decode_frame_props failed\n");

                goto error;

            }



            for (i = 0; i < 2; i++) {

                CUDA_MEMCPY2D cpy = {

                    .srcMemoryType = CU_MEMORYTYPE_DEVICE,

                    .dstMemoryType = CU_MEMORYTYPE_DEVICE,

                    .srcDevice     = mapped_frame,

                    .dstDevice     = (CUdeviceptr)frame->data[i],

                    .srcPitch      = pitch,

                    .dstPitch      = frame->linesize[i],

                    .srcY          = offset,

                    .WidthInBytes  = FFMIN(pitch, frame->linesize[i]),

                    .Height        = avctx->coded_height >> (i ? 1 : 0),

                };



                ret = CHECK_CU(cuMemcpy2D(&cpy));

                if (ret < 0)

                    goto error;



                offset += avctx->coded_height;

            }

        } else if (avctx->pix_fmt == AV_PIX_FMT_NV12) {

            AVFrame *tmp_frame = av_frame_alloc();

            if (!tmp_frame) {

                av_log(avctx, AV_LOG_ERROR, "av_frame_alloc failed\n");

                ret = AVERROR(ENOMEM);

                goto error;

            }



            tmp_frame->format        = AV_PIX_FMT_CUDA;

            tmp_frame->hw_frames_ctx = av_buffer_ref(ctx->hwframe);

            tmp_frame->data[0]       = (uint8_t*)mapped_frame;

            tmp_frame->linesize[0]   = pitch;

            tmp_frame->data[1]       = (uint8_t*)(mapped_frame + avctx->coded_height * pitch);

            tmp_frame->linesize[1]   = pitch;

            tmp_frame->width         = avctx->width;

            tmp_frame->height        = avctx->height;



            ret = ff_get_buffer(avctx, frame, 0);

            if (ret < 0) {

                av_log(avctx, AV_LOG_ERROR, "ff_get_buffer failed\n");

                av_frame_free(&tmp_frame);

                goto error;

            }



            ret = av_hwframe_transfer_data(frame, tmp_frame, 0);

            if (ret) {

                av_log(avctx, AV_LOG_ERROR, "av_hwframe_transfer_data failed\n");

                av_frame_free(&tmp_frame);

                goto error;

            }



            av_frame_free(&tmp_frame);

        } else {

            ret = AVERROR_BUG;

            goto error;

        }



        frame->width = avctx->width;

        frame->height = avctx->height;

        if (avctx->pkt_timebase.num && avctx->pkt_timebase.den)

            frame->pts = av_rescale_q(dispinfo.timestamp, (AVRational){1, 10000000}, avctx->pkt_timebase);

        else

            frame->pts = dispinfo.timestamp;



        /* CUVIDs opaque reordering breaks the internal pkt logic.

         * So set pkt_pts and clear all the other pkt_ fields.

         */

        frame->pkt_pts = frame->pts;

        av_frame_set_pkt_pos(frame, -1);

        av_frame_set_pkt_duration(frame, 0);

        av_frame_set_pkt_size(frame, -1);



        frame->interlaced_frame = !dispinfo.progressive_frame;



        if (!dispinfo.progressive_frame)

            frame->top_field_first = dispinfo.top_field_first;



        *got_frame = 1;

    } else {

        *got_frame = 0;

    }



error:

    if (mapped_frame)

        eret = CHECK_CU(cuvidUnmapVideoFrame(ctx->cudecoder, mapped_frame));



    eret = CHECK_CU(cuCtxPopCurrent(&dummy));



    if (eret < 0)

        return eret;

    else

        return ret;

}
