static int decode_audio(InputStream *ist, AVPacket *pkt, int *got_output)

{

    AVFrame *decoded_frame, *f;

    AVCodecContext *avctx = ist->dec_ctx;

    int i, ret, err = 0;



    if (!ist->decoded_frame && !(ist->decoded_frame = av_frame_alloc()))

        return AVERROR(ENOMEM);

    if (!ist->filter_frame && !(ist->filter_frame = av_frame_alloc()))

        return AVERROR(ENOMEM);

    decoded_frame = ist->decoded_frame;



    ret = decode(avctx, decoded_frame, got_output, pkt);

    if (!*got_output || ret < 0)

        return ret;



    ist->samples_decoded += decoded_frame->nb_samples;

    ist->frames_decoded++;



    /* if the decoder provides a pts, use it instead of the last packet pts.

       the decoder could be delaying output by a packet or more. */

    if (decoded_frame->pts != AV_NOPTS_VALUE)

        ist->next_dts = decoded_frame->pts;

    else if (pkt && pkt->pts != AV_NOPTS_VALUE) {

        decoded_frame->pts = pkt->pts;

    }



    if (decoded_frame->pts != AV_NOPTS_VALUE)

        decoded_frame->pts = av_rescale_q(decoded_frame->pts,

                                          ist->st->time_base,

                                          (AVRational){1, avctx->sample_rate});

    ist->nb_samples = decoded_frame->nb_samples;

    for (i = 0; i < ist->nb_filters; i++) {

        if (i < ist->nb_filters - 1) {

            f = ist->filter_frame;

            err = av_frame_ref(f, decoded_frame);

            if (err < 0)

                break;

        } else

            f = decoded_frame;



        err = ifilter_send_frame(ist->filters[i], f);

        if (err < 0)

            break;

    }



    av_frame_unref(ist->filter_frame);

    av_frame_unref(decoded_frame);

    return err < 0 ? err : ret;

}
