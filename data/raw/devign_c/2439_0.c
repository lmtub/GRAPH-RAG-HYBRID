static int pulse_set_volume(PulseData *s, double volume)

{

    pa_operation *op;

    pa_cvolume cvol;

    pa_volume_t vol;

    const pa_sample_spec *ss = pa_stream_get_sample_spec(s->stream);



    vol = pa_sw_volume_multiply(lround(volume * PA_VOLUME_NORM), s->base_volume);

    pa_cvolume_set(&cvol, ss->channels, PA_VOLUME_NORM);

    pa_sw_cvolume_multiply_scalar(&cvol, &cvol, vol);

    pa_threaded_mainloop_lock(s->mainloop);

    op = pa_context_set_sink_input_volume(s->ctx, pa_stream_get_index(s->stream),

                                          &cvol, pulse_context_result, s);

    return pulse_finish_context_operation(s, op, "pa_context_set_sink_input_volume");

}
