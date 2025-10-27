static int compand_nodelay(AVFilterContext *ctx, AVFrame *frame)

{

    CompandContext *s = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];

    const int channels = inlink->channels;

    const int nb_samples = frame->nb_samples;

    AVFrame *out_frame;

    int chan, i;



    if (av_frame_is_writable(frame)) {

        out_frame = frame;

    } else {

        out_frame = ff_get_audio_buffer(inlink, nb_samples);

        if (!out_frame)

            return AVERROR(ENOMEM);

        av_frame_copy_props(out_frame, frame);

    }



    for (chan = 0; chan < channels; chan++) {

        const double *src = (double *)frame->extended_data[chan];

        double *dst = (double *)out_frame->extended_data[chan];

        ChanParam *cp = &s->channels[chan];



        for (i = 0; i < nb_samples; i++) {

            update_volume(cp, fabs(src[i]));



            dst[i] = av_clipd(src[i] * get_volume(s, cp->volume), -1, 1);

        }

    }



    if (frame != out_frame)

        av_frame_free(&frame);



    return ff_filter_frame(ctx->outputs[0], out_frame);

}
