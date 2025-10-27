static int audio_attach_capture (HWVoiceOut *hw)

{

    AudioState *s = &glob_audio_state;

    CaptureVoiceOut *cap;



    audio_detach_capture (hw);

    for (cap = s->cap_head.lh_first; cap; cap = cap->entries.le_next) {

        SWVoiceCap *sc;

        SWVoiceOut *sw;

        HWVoiceOut *hw_cap = &cap->hw;



        sc = audio_calloc (AUDIO_FUNC, 1, sizeof (*sc));

        if (!sc) {

            dolog ("Could not allocate soft capture voice (%zu bytes)\n",

                   sizeof (*sc));

            return -1;

        }



        sc->cap = cap;

        sw = &sc->sw;

        sw->hw = hw_cap;

        sw->info = hw->info;

        sw->empty = 1;

        sw->active = hw->enabled;

        sw->conv = noop_conv;

        sw->ratio = ((int64_t) hw_cap->info.freq << 32) / sw->info.freq;

        sw->rate = st_rate_start (sw->info.freq, hw_cap->info.freq);

        if (!sw->rate) {

            dolog ("Could not start rate conversion for `%s'\n", SW_NAME (sw));

            qemu_free (sw);

            return -1;

        }

        LIST_INSERT_HEAD (&hw_cap->sw_head, sw, entries);

        LIST_INSERT_HEAD (&hw->cap_head, sc, entries);

#ifdef DEBUG_CAPTURE

        asprintf (&sw->name, "for %p %d,%d,%d",

                  hw, sw->info.freq, sw->info.bits, sw->info.nchannels);

        dolog ("Added %s active = %d\n", sw->name, sw->active);

#endif

        if (sw->active) {

            audio_capture_maybe_changed (cap, 1);

        }

    }

    return 0;

}
