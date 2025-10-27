static int request_samples(AVFilterContext *ctx, int min_samples)

{

    MixContext *s = ctx->priv;

    int i, ret;



    av_assert0(s->nb_inputs > 1);



    for (i = 1; i < s->nb_inputs; i++) {

        ret = 0;

        if (!(s->input_state[i] & INPUT_ON))

            continue;

        if (av_audio_fifo_size(s->fifos[i]) >= min_samples)

            continue;

        ret = ff_request_frame(ctx->inputs[i]);

        if (ret == AVERROR_EOF) {

            s->input_state[i] |= INPUT_EOF;

            if (av_audio_fifo_size(s->fifos[i]) == 0) {

                s->input_state[i] = 0;

                continue;

            }

        } else if (ret < 0)

            return ret;

    }

    return output_frame(ctx->outputs[0], 1);

}
