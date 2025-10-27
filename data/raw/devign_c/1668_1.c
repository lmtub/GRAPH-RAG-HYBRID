static int push_samples(AVFilterContext *ctx, int nb_samples)

{

    AVFilterLink *outlink = ctx->outputs[0];

    LoopContext *s = ctx->priv;

    AVFrame *out;

    int ret, i = 0;



    while (s->loop != 0 && i < nb_samples) {

        out = ff_get_audio_buffer(outlink, FFMIN(nb_samples, s->nb_samples - s->current_sample));

        if (!out)

            return AVERROR(ENOMEM);

        ret = av_audio_fifo_peek_at(s->fifo, (void **)out->extended_data, out->nb_samples, s->current_sample);

        if (ret < 0)

            return ret;

        out->pts = s->pts;

        out->nb_samples = ret;

        s->pts += out->nb_samples;

        i += out->nb_samples;

        s->current_sample += out->nb_samples;



        ret = ff_filter_frame(outlink, out);

        if (ret < 0)

            return ret;



        if (s->current_sample >= s->nb_samples) {

            s->current_sample = 0;



            if (s->loop > 0)

                s->loop--;

        }

    }



    return ret;

}
