static int rate_get_samples (struct audio_pcm_info *info, SpiceRateCtl *rate)

{

    int64_t now;

    int64_t ticks;

    int64_t bytes;

    int64_t samples;



    now = qemu_get_clock (vm_clock);

    ticks = now - rate->start_ticks;

    bytes = muldiv64 (ticks, info->bytes_per_second, get_ticks_per_sec ());

    samples = (bytes - rate->bytes_sent) >> info->shift;

    if (samples < 0 || samples > 65536) {

        fprintf (stderr, "Resetting rate control (%" PRId64 " samples)\n", samples);

        rate_start (rate);

        samples = 0;

    }

    rate->bytes_sent += samples << info->shift;

    return samples;

}
