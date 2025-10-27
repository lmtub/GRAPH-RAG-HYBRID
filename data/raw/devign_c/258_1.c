static int ffmmal_read_frame(AVCodecContext *avctx, AVFrame *frame, int *got_frame)

{

    MMALDecodeContext *ctx = avctx->priv_data;

    MMAL_BUFFER_HEADER_T *buffer = NULL;

    MMAL_STATUS_T status = 0;

    int ret = 0;



    if (ctx->eos_received)

        goto done;



    while (1) {

        // To ensure decoding in lockstep with a constant delay between fed packets

        // and output frames, we always wait until an output buffer is available.

        // Except during start we don't know after how many input packets the decoder

        // is going to return the first buffer, and we can't distinguish decoder

        // being busy from decoder waiting for input. So just poll at the start and

        // keep feeding new data to the buffer.

        // We are pretty sure the decoder will produce output if we sent more input

        // frames than what a h264 decoder could logically delay. This avoids too

        // excessive buffering.

        // We also wait if we sent eos, but didn't receive it yet (think of decoding

        // stream with a very low number of frames).

        if (ctx->frames_output || ctx->packets_sent > MAX_DELAYED_FRAMES ||

            (ctx->packets_sent && ctx->eos_sent)) {

            // MMAL will ignore broken input packets, which means the frame we

            // expect here may never arrive. Dealing with this correctly is

            // complicated, so here's a hack to avoid that it freezes forever

            // in this unlikely situation.

            buffer = mmal_queue_timedwait(ctx->queue_decoded_frames, 100);

            if (!buffer) {

                av_log(avctx, AV_LOG_ERROR, "Did not get output frame from MMAL.\n");

                ret = AVERROR_UNKNOWN;

                goto done;

            }

        } else {

            buffer = mmal_queue_get(ctx->queue_decoded_frames);

            if (!buffer)

                goto done;

        }



        ctx->eos_received |= !!(buffer->flags & MMAL_BUFFER_HEADER_FLAG_EOS);

        if (ctx->eos_received)

            goto done;



        if (buffer->cmd == MMAL_EVENT_FORMAT_CHANGED) {

            MMAL_COMPONENT_T *decoder = ctx->decoder;

            MMAL_EVENT_FORMAT_CHANGED_T *ev = mmal_event_format_changed_get(buffer);

            MMAL_BUFFER_HEADER_T *stale_buffer;



            av_log(avctx, AV_LOG_INFO, "Changing output format.\n");



            if ((status = mmal_port_disable(decoder->output[0])))

                goto done;



            while ((stale_buffer = mmal_queue_get(ctx->queue_decoded_frames)))

                mmal_buffer_header_release(stale_buffer);



            mmal_format_copy(decoder->output[0]->format, ev->format);



            if ((ret = ffmal_update_format(avctx)) < 0)

                goto done;



            if ((status = mmal_port_enable(decoder->output[0], output_callback)))

                goto done;



            if ((ret = ffmmal_fill_output_port(avctx)) < 0)

                goto done;



            if ((ret = ffmmal_fill_input_port(avctx)) < 0)

                goto done;



            mmal_buffer_header_release(buffer);

            continue;

        } else if (buffer->cmd) {

            char s[20];

            av_get_codec_tag_string(s, sizeof(s), buffer->cmd);

            av_log(avctx, AV_LOG_WARNING, "Unknown MMAL event %s on output port\n", s);

            goto done;

        } else if (buffer->length == 0) {

            // Unused output buffer that got drained after format change.

            mmal_buffer_header_release(buffer);

            continue;

        }



        ctx->frames_output++;



        if ((ret = ffmal_copy_frame(avctx, frame, buffer)) < 0)

            goto done;



        *got_frame = 1;

        break;

    }



done:

    if (buffer)

        mmal_buffer_header_release(buffer);

    if (status && ret >= 0)

        ret = AVERROR_UNKNOWN;

    return ret;

}
