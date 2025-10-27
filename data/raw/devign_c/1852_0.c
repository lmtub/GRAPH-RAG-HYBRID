static int filter_frame(AVFilterLink *inlink, AVFrame *insamples)

{

    AVFilterContext *ctx = inlink->dst;

    ASNSContext *asns = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    int ret;

    int nb_samples = insamples->nb_samples;



    if (av_audio_fifo_space(asns->fifo) < nb_samples) {

        av_log(ctx, AV_LOG_DEBUG, "No space for %d samples, stretching audio fifo\n", nb_samples);

        ret = av_audio_fifo_realloc(asns->fifo, av_audio_fifo_size(asns->fifo) + nb_samples);

        if (ret < 0) {

            av_log(ctx, AV_LOG_ERROR,

                   "Stretching audio fifo failed, discarded %d samples\n", nb_samples);

            return -1;

        }

    }

    av_audio_fifo_write(asns->fifo, (void **)insamples->extended_data, nb_samples);

    if (asns->next_out_pts == AV_NOPTS_VALUE)

        asns->next_out_pts = insamples->pts;

    av_frame_free(&insamples);



    while (av_audio_fifo_size(asns->fifo) >= asns->nb_out_samples)

        push_samples(outlink);

    return 0;

}
