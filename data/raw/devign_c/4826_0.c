static void audio_reset_timer (AudioState *s)

{

    if (audio_is_timer_needed ()) {

        timer_mod (s->ts,

            qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + conf.period.ticks);

    }

    else {

        timer_del (s->ts);

    }

}
