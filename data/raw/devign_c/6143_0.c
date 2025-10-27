static int encode_frame(AVCodecContext* avc_context, uint8_t *outbuf,

                        int buf_size, void *data)

{

    th_ycbcr_buffer t_yuv_buffer;

    TheoraContext *h = avc_context->priv_data;

    AVFrame *frame = data;

    ogg_packet o_packet;

    int result, i;



    // EOS, finish and get 1st pass stats if applicable

    if (!frame) {

        th_encode_packetout(h->t_state, 1, &o_packet);

        if (avc_context->flags & CODEC_FLAG_PASS1)

            if (get_stats(avc_context, 1))

                return -1;

        return 0;

    }



    /* Copy planes to the theora yuv_buffer */

    for (i = 0; i < 3; i++) {

        t_yuv_buffer[i].width  = FFALIGN(avc_context->width,  16) >> (i && h->uv_hshift);

        t_yuv_buffer[i].height = FFALIGN(avc_context->height, 16) >> (i && h->uv_vshift);

        t_yuv_buffer[i].stride = frame->linesize[i];

        t_yuv_buffer[i].data   = frame->data[i];

    }



    if (avc_context->flags & CODEC_FLAG_PASS2)

        if (submit_stats(avc_context))

            return -1;



    /* Now call into theora_encode_YUVin */

    result = th_encode_ycbcr_in(h->t_state, t_yuv_buffer);

    if (result) {

        const char* message;

        switch (result) {

        case -1:

            message = "differing frame sizes";

            break;

        case TH_EINVAL:

            message = "encoder is not ready or is finished";

            break;

        default:

            message = "unknown reason";

            break;

        }

        av_log(avc_context, AV_LOG_ERROR, "theora_encode_YUVin failed (%s) [%d]\n", message, result);

        return -1;

    }



    if (avc_context->flags & CODEC_FLAG_PASS1)

        if (get_stats(avc_context, 0))

            return -1;



    /* Pick up returned ogg_packet */

    result = th_encode_packetout(h->t_state, 0, &o_packet);

    switch (result) {

    case 0:

        /* No packet is ready */

        return 0;

    case 1:

        /* Success, we have a packet */

        break;

    default:

        av_log(avc_context, AV_LOG_ERROR, "theora_encode_packetout failed [%d]\n", result);

        return -1;

    }



    /* Copy ogg_packet content out to buffer */

    if (buf_size < o_packet.bytes) {

        av_log(avc_context, AV_LOG_ERROR, "encoded frame too large\n");

        return -1;

    }

    memcpy(outbuf, o_packet.packet, o_packet.bytes);



    // HACK: does not take codec delay into account (neither does the decoder though)

    avc_context->coded_frame->pts = frame->pts;

    avc_context->coded_frame->key_frame = !(o_packet.granulepos & h->keyframe_mask);



    return o_packet.bytes;

}
