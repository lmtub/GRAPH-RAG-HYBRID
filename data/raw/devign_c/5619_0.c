static void pulse_get_output_timestamp(AVFormatContext *h, int stream, int64_t *dts, int64_t *wall)

{

    PulseData *s = h->priv_data;

    pa_usec_t latency;

    int neg;

    pa_threaded_mainloop_lock(s->mainloop);

    pa_stream_get_latency(s->stream, &latency, &neg);

    pa_threaded_mainloop_unlock(s->mainloop);

    *wall = av_gettime();

    *dts = s->timestamp - (neg ? -latency : latency);

}
