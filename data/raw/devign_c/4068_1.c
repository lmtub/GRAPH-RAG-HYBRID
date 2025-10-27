static av_cold void uninit(AVFilterContext *ctx)

{

    DynamicAudioNormalizerContext *s = ctx->priv;

    int c;



    av_freep(&s->prev_amplification_factor);

    av_freep(&s->dc_correction_value);

    av_freep(&s->compress_threshold);

    av_freep(&s->fade_factors[0]);

    av_freep(&s->fade_factors[1]);



    for (c = 0; c < s->channels; c++) {

        cqueue_free(s->gain_history_original[c]);

        cqueue_free(s->gain_history_minimum[c]);

        cqueue_free(s->gain_history_smoothed[c]);

    }



    av_freep(&s->gain_history_original);

    av_freep(&s->gain_history_minimum);

    av_freep(&s->gain_history_smoothed);



    av_freep(&s->weights);



    ff_bufqueue_discard_all(&s->queue);

}
