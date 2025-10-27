static int output_frame(AVFilterLink *outlink, int need_request)

{

    AVFilterContext *ctx = outlink->src;

    MixContext      *s = ctx->priv;

    AVFrame *out_buf, *in_buf;

    int nb_samples, ns, ret, i;



    ret = calc_active_inputs(s);

    if (ret < 0)

        return ret;



    if (s->input_state[0] & INPUT_ON) {

        /* first input live: use the corresponding frame size */

        nb_samples = frame_list_next_frame_size(s->frame_list);

        for (i = 1; i < s->nb_inputs; i++) {

            if (s->input_state[i] & INPUT_ON) {

                ns = av_audio_fifo_size(s->fifos[i]);

                if (ns < nb_samples) {

                    if (!(s->input_state[i] & INPUT_EOF))

                        /* unclosed input with not enough samples */

                        return need_request ? ff_request_frame(ctx->inputs[i]) : 0;

                    /* closed input to drain */

                    nb_samples = ns;

                }

            }

        }

    } else {

        /* first input closed: use the available samples */

        nb_samples = INT_MAX;

        for (i = 1; i < s->nb_inputs; i++) {

            if (s->input_state[i] & INPUT_ON) {

                ns = av_audio_fifo_size(s->fifos[i]);

                nb_samples = FFMIN(nb_samples, ns);

            }

        }

        if (nb_samples == INT_MAX)

            return AVERROR_EOF;

    }



    s->next_pts = frame_list_next_pts(s->frame_list);

    frame_list_remove_samples(s->frame_list, nb_samples);



    calculate_scales(s, nb_samples);



    if (nb_samples == 0)

        return 0;



    out_buf = ff_get_audio_buffer(outlink, nb_samples);

    if (!out_buf)

        return AVERROR(ENOMEM);



    in_buf = ff_get_audio_buffer(outlink, nb_samples);

    if (!in_buf) {

        av_frame_free(&out_buf);

        return AVERROR(ENOMEM);

    }



    for (i = 0; i < s->nb_inputs; i++) {

        if (s->input_state[i] & INPUT_ON) {

            int planes, plane_size, p;



            av_audio_fifo_read(s->fifos[i], (void **)in_buf->extended_data,

                               nb_samples);



            planes     = s->planar ? s->nb_channels : 1;

            plane_size = nb_samples * (s->planar ? 1 : s->nb_channels);

            plane_size = FFALIGN(plane_size, 16);



            if (out_buf->format == AV_SAMPLE_FMT_FLT ||

                out_buf->format == AV_SAMPLE_FMT_FLTP) {

                for (p = 0; p < planes; p++) {

                    s->fdsp->vector_fmac_scalar((float *)out_buf->extended_data[p],

                                                (float *) in_buf->extended_data[p],

                                                s->input_scale[i], plane_size);

                }

            } else {

                for (p = 0; p < planes; p++) {

                    s->fdsp->vector_dmac_scalar((double *)out_buf->extended_data[p],

                                                (double *) in_buf->extended_data[p],

                                                s->input_scale[i], plane_size);

                }

            }

        }

    }

    av_frame_free(&in_buf);



    out_buf->pts = s->next_pts;

    if (s->next_pts != AV_NOPTS_VALUE)

        s->next_pts += nb_samples;



    return ff_filter_frame(outlink, out_buf);

}
