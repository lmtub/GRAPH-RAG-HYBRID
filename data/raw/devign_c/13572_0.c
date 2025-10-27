static int omx_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                            const AVFrame *frame, int *got_packet)

{

    OMXCodecContext *s = avctx->priv_data;

    int ret = 0;

    OMX_BUFFERHEADERTYPE* buffer;

    OMX_ERRORTYPE err;



    if (frame) {

        uint8_t *dst[4];

        int linesize[4];

        int need_copy;

        buffer = get_buffer(&s->input_mutex, &s->input_cond,

                            &s->num_free_in_buffers, s->free_in_buffers, 1);



        buffer->nFilledLen = av_image_fill_arrays(dst, linesize, buffer->pBuffer, avctx->pix_fmt, s->stride, s->plane_size, 1);



        if (s->input_zerocopy) {

            uint8_t *src[4] = { NULL };

            int src_linesize[4];

            av_image_fill_arrays(src, src_linesize, frame->data[0], avctx->pix_fmt, s->stride, s->plane_size, 1);

            if (frame->linesize[0] == src_linesize[0] &&

                frame->linesize[1] == src_linesize[1] &&

                frame->linesize[2] == src_linesize[2] &&

                frame->data[1] == src[1] &&

                frame->data[2] == src[2]) {

                // If the input frame happens to have all planes stored contiguously,

                // with the right strides, just clone the frame and set the OMX

                // buffer header to point to it

                AVFrame *local = av_frame_clone(frame);

                if (!local) {

                    // Return the buffer to the queue so it's not lost

                    append_buffer(&s->input_mutex, &s->input_cond, &s->num_free_in_buffers, s->free_in_buffers, buffer);

                    return AVERROR(ENOMEM);

                } else {

                    buffer->pAppPrivate = local;

                    buffer->pOutputPortPrivate = NULL;

                    buffer->pBuffer = local->data[0];

                    need_copy = 0;

                }

            } else {

                // If not, we need to allocate a new buffer with the right

                // size and copy the input frame into it.

                uint8_t *buf = av_malloc(av_image_get_buffer_size(avctx->pix_fmt, s->stride, s->plane_size, 1));

                if (!buf) {

                    // Return the buffer to the queue so it's not lost

                    append_buffer(&s->input_mutex, &s->input_cond, &s->num_free_in_buffers, s->free_in_buffers, buffer);

                    return AVERROR(ENOMEM);

                } else {

                    buffer->pAppPrivate = buf;

                    // Mark that pAppPrivate is an av_malloc'ed buffer, not an AVFrame

                    buffer->pOutputPortPrivate = (void*) 1;

                    buffer->pBuffer = buf;

                    need_copy = 1;

                    buffer->nFilledLen = av_image_fill_arrays(dst, linesize, buffer->pBuffer, avctx->pix_fmt, s->stride, s->plane_size, 1);

                }

            }

        } else {

            need_copy = 1;

        }

        if (need_copy)

            av_image_copy(dst, linesize, (const uint8_t**) frame->data, frame->linesize, avctx->pix_fmt, avctx->width, avctx->height);

        buffer->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;

        buffer->nOffset = 0;

        // Convert the timestamps to microseconds; some encoders can ignore

        // the framerate and do VFR bit allocation based on timestamps.

        buffer->nTimeStamp = to_omx_ticks(av_rescale_q(frame->pts, avctx->time_base, AV_TIME_BASE_Q));

        err = OMX_EmptyThisBuffer(s->handle, buffer);

        if (err != OMX_ErrorNone) {

            append_buffer(&s->input_mutex, &s->input_cond, &s->num_free_in_buffers, s->free_in_buffers, buffer);

            av_log(avctx, AV_LOG_ERROR, "OMX_EmptyThisBuffer failed: %x\n", err);

            return AVERROR_UNKNOWN;

        }

        s->num_in_frames++;

    }



    while (!*got_packet && ret == 0) {

        // Only wait for output if flushing and not all frames have been output

        buffer = get_buffer(&s->output_mutex, &s->output_cond,

                            &s->num_done_out_buffers, s->done_out_buffers,

                            !frame && s->num_out_frames < s->num_in_frames);

        if (!buffer)

            break;



        if (buffer->nFlags & OMX_BUFFERFLAG_CODECCONFIG && avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

            if ((ret = av_reallocp(&avctx->extradata, avctx->extradata_size + buffer->nFilledLen + AV_INPUT_BUFFER_PADDING_SIZE)) < 0) {

                avctx->extradata_size = 0;

                goto end;

            }

            memcpy(avctx->extradata + avctx->extradata_size, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);

            avctx->extradata_size += buffer->nFilledLen;

            memset(avctx->extradata + avctx->extradata_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

        } else {

            if (buffer->nFlags & OMX_BUFFERFLAG_ENDOFFRAME)

                s->num_out_frames++;

            if (!(buffer->nFlags & OMX_BUFFERFLAG_ENDOFFRAME) || !pkt->data) {

                // If the output packet isn't preallocated, just concatenate everything in our

                // own buffer

                int newsize = s->output_buf_size + buffer->nFilledLen + AV_INPUT_BUFFER_PADDING_SIZE;

                if ((ret = av_reallocp(&s->output_buf, newsize)) < 0) {

                    s->output_buf_size = 0;

                    goto end;

                }

                memcpy(s->output_buf + s->output_buf_size, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);

                s->output_buf_size += buffer->nFilledLen;

                if (buffer->nFlags & OMX_BUFFERFLAG_ENDOFFRAME) {

                    if ((ret = av_packet_from_data(pkt, s->output_buf, s->output_buf_size)) < 0) {

                        av_freep(&s->output_buf);

                        s->output_buf_size = 0;

                        goto end;

                    }

                    s->output_buf = NULL;

                    s->output_buf_size = 0;

                }

            } else {

                // End of frame, and the caller provided a preallocated frame

                if ((ret = ff_alloc_packet2(avctx, pkt, s->output_buf_size + buffer->nFilledLen, 0)) < 0) {

                    av_log(avctx, AV_LOG_ERROR, "Error getting output packet of size %d.\n",

                           (int)(s->output_buf_size + buffer->nFilledLen));

                    goto end;

                }

                memcpy(pkt->data, s->output_buf, s->output_buf_size);

                memcpy(pkt->data + s->output_buf_size, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);

                av_freep(&s->output_buf);

                s->output_buf_size = 0;

            }

            if (buffer->nFlags & OMX_BUFFERFLAG_ENDOFFRAME) {

                pkt->pts = av_rescale_q(from_omx_ticks(buffer->nTimeStamp), AV_TIME_BASE_Q, avctx->time_base);

                // We don't currently enable B-frames for the encoders, so set

                // pkt->dts = pkt->pts. (The calling code behaves worse if the encoder

                // doesn't set the dts).

                pkt->dts = pkt->pts;

                if (buffer->nFlags & OMX_BUFFERFLAG_SYNCFRAME)

                    pkt->flags |= AV_PKT_FLAG_KEY;

                *got_packet = 1;

            }

        }

end:

        err = OMX_FillThisBuffer(s->handle, buffer);

        if (err != OMX_ErrorNone) {

            append_buffer(&s->output_mutex, &s->output_cond, &s->num_done_out_buffers, s->done_out_buffers, buffer);

            av_log(avctx, AV_LOG_ERROR, "OMX_FillThisBuffer failed: %x\n", err);

            ret = AVERROR_UNKNOWN;

        }

    }

    return ret;

}
