static int vtenc_cm_to_avpacket(

    AVCodecContext    *avctx,

    CMSampleBufferRef sample_buffer,

    AVPacket          *pkt,

    ExtraSEI          *sei)

{

    VTEncContext *vtctx = avctx->priv_data;



    int     status;

    bool    is_key_frame;

    bool    add_header;

    size_t  length_code_size;

    size_t  header_size = 0;

    size_t  in_buf_size;

    size_t  out_buf_size;

    size_t  sei_nalu_size = 0;

    int64_t dts_delta;

    int64_t time_base_num;

    int nalu_count;

    CMTime  pts;

    CMTime  dts;

    CMVideoFormatDescriptionRef vid_fmt;





    vtenc_get_frame_info(sample_buffer, &is_key_frame);

    status = get_length_code_size(avctx, sample_buffer, &length_code_size);

    if (status) return status;



    add_header = is_key_frame && !(avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER);



    if (add_header) {

        vid_fmt = CMSampleBufferGetFormatDescription(sample_buffer);

        if (!vid_fmt) {

            av_log(avctx, AV_LOG_ERROR, "Cannot get format description.\n");

            return AVERROR_EXTERNAL;

        }



        int status = get_params_size(avctx, vid_fmt, &header_size);

        if (status) return status;

    }



    status = count_nalus(length_code_size, sample_buffer, &nalu_count);

    if(status)

        return status;



    if (sei) {

        sei_nalu_size = sizeof(start_code) + 3 + sei->size + 1;

    }



    in_buf_size = CMSampleBufferGetTotalSampleSize(sample_buffer);

    out_buf_size = header_size +

                   in_buf_size +

                   sei_nalu_size +

                   nalu_count * ((int)sizeof(start_code) - (int)length_code_size);



    status = ff_alloc_packet2(avctx, pkt, out_buf_size, out_buf_size);

    if (status < 0)

        return status;



    if (add_header) {

        status = copy_param_sets(avctx, vid_fmt, pkt->data, out_buf_size);

        if(status) return status;

    }



    status = copy_replace_length_codes(

        avctx,

        length_code_size,

        sample_buffer,

        pkt->data + header_size,

        pkt->size - header_size - sei_nalu_size

    );



    if (status) {

        av_log(avctx, AV_LOG_ERROR, "Error copying packet data: %d", status);

        return status;

    }



    if (sei_nalu_size > 0) {

        uint8_t *sei_nalu = pkt->data + pkt->size - sei_nalu_size;

        memcpy(sei_nalu, start_code, sizeof(start_code));

        sei_nalu += sizeof(start_code);

        sei_nalu[0] = H264_NAL_SEI;

        sei_nalu[1] = SEI_TYPE_USER_DATA_REGISTERED;

        sei_nalu[2] = sei->size;

        sei_nalu += 3;

        memcpy(sei_nalu, sei->data, sei->size);

        sei_nalu += sei->size;

        sei_nalu[0] = 1; // RBSP

    }



    if (is_key_frame) {

        pkt->flags |= AV_PKT_FLAG_KEY;

    }



    pts = CMSampleBufferGetPresentationTimeStamp(sample_buffer);

    dts = CMSampleBufferGetDecodeTimeStamp      (sample_buffer);



    if (CMTIME_IS_INVALID(dts)) {

        if (!vtctx->has_b_frames) {

            dts = pts;

        } else {

            av_log(avctx, AV_LOG_ERROR, "DTS is invalid.\n");

            return AVERROR_EXTERNAL;

        }

    }



    dts_delta = vtctx->dts_delta >= 0 ? vtctx->dts_delta : 0;

    time_base_num = avctx->time_base.num;

    pkt->pts = pts.value / time_base_num;

    pkt->dts = dts.value / time_base_num - dts_delta;

    pkt->size = out_buf_size;



    return 0;

}
