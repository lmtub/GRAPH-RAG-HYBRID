static int no_run_out (HWVoiceOut *hw, int live)

{

    NoVoiceOut *no = (NoVoiceOut *) hw;

    int decr, samples;

    int64_t now;

    int64_t ticks;

    int64_t bytes;



    now = qemu_get_clock (vm_clock);

    ticks = now - no->old_ticks;

    bytes = muldiv64 (ticks, hw->info.bytes_per_second, get_ticks_per_sec ());

    bytes = audio_MIN (bytes, INT_MAX);

    samples = bytes >> hw->info.shift;



    no->old_ticks = now;

    decr = audio_MIN (live, samples);

    hw->rpos = (hw->rpos + decr) % hw->samples;

    return decr;

}
