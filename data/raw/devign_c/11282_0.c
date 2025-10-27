static av_cold int omx_encode_init(AVCodecContext *avctx)

{

    OMXCodecContext *s = avctx->priv_data;

    int ret = AVERROR_ENCODER_NOT_FOUND;

    const char *role;

    OMX_BUFFERHEADERTYPE *buffer;

    OMX_ERRORTYPE err;



#if CONFIG_OMX_RPI

    s->input_zerocopy = 1;

#endif



    s->omx_context = omx_init(avctx, s->libname, s->libprefix);

    if (!s->omx_context)

        return AVERROR_ENCODER_NOT_FOUND;



    pthread_mutex_init(&s->state_mutex, NULL);

    pthread_cond_init(&s->state_cond, NULL);

    pthread_mutex_init(&s->input_mutex, NULL);

    pthread_cond_init(&s->input_cond, NULL);

    pthread_mutex_init(&s->output_mutex, NULL);

    pthread_cond_init(&s->output_cond, NULL);

    s->mutex_cond_inited = 1;

    s->avctx = avctx;

    s->state = OMX_StateLoaded;

    s->error = OMX_ErrorNone;



    switch (avctx->codec->id) {

    case AV_CODEC_ID_MPEG4:

        role = "video_encoder.mpeg4";

        break;

    case AV_CODEC_ID_H264:

        role = "video_encoder.avc";

        break;

    default:

        return AVERROR(ENOSYS);

    }



    if ((ret = find_component(s->omx_context, avctx, role, s->component_name, sizeof(s->component_name))) < 0)

        goto fail;



    av_log(avctx, AV_LOG_INFO, "Using %s\n", s->component_name);



    if ((ret = omx_component_init(avctx, role)) < 0)

        goto fail;



    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

        while (1) {

            buffer = get_buffer(&s->output_mutex, &s->output_cond,

                                &s->num_done_out_buffers, s->done_out_buffers, 1);

            if (buffer->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {

                if ((ret = av_reallocp(&avctx->extradata, avctx->extradata_size + buffer->nFilledLen + AV_INPUT_BUFFER_PADDING_SIZE)) < 0) {

                    avctx->extradata_size = 0;

                    goto fail;

                }

                memcpy(avctx->extradata + avctx->extradata_size, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);

                avctx->extradata_size += buffer->nFilledLen;

                memset(avctx->extradata + avctx->extradata_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

            }

            err = OMX_FillThisBuffer(s->handle, buffer);

            if (err != OMX_ErrorNone) {

                append_buffer(&s->output_mutex, &s->output_cond,

                              &s->num_done_out_buffers, s->done_out_buffers, buffer);

                av_log(avctx, AV_LOG_ERROR, "OMX_FillThisBuffer failed: %x\n", err);

                ret = AVERROR_UNKNOWN;

                goto fail;

            }

            if (avctx->codec->id == AV_CODEC_ID_H264) {

                // For H.264, the extradata can be returned in two separate buffers

                // (the videocore encoder on raspberry pi does this);

                // therefore check that we have got both SPS and PPS before continuing.

                int nals[32] = { 0 };

                int i;

                for (i = 0; i + 4 < avctx->extradata_size; i++) {

                     if (!avctx->extradata[i + 0] &&

                         !avctx->extradata[i + 1] &&

                         !avctx->extradata[i + 2] &&

                         avctx->extradata[i + 3] == 1) {

                         nals[avctx->extradata[i + 4] & 0x1f]++;

                     }

                }

                if (nals[NAL_SPS] && nals[NAL_PPS])

                    break;

            } else {

                if (avctx->extradata_size > 0)

                    break;

            }

        }

    }



    return 0;

fail:

    return ret;

}
