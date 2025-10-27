static int hda_audio_post_load(void *opaque, int version)

{

    HDAAudioState *a = opaque;

    HDAAudioStream *st;

    int i;



    dprint(a, 1, "%s\n", __FUNCTION__);

    if (version == 1) {

        /* assume running_compat[] is for output streams */

        for (i = 0; i < ARRAY_SIZE(a->running_compat); i++)

            a->running_real[16 + i] = a->running_compat[i];

    }



    for (i = 0; i < ARRAY_SIZE(a->st); i++) {

        st = a->st + i;

        if (st->node == NULL)

            continue;

        hda_codec_parse_fmt(st->format, &st->as);

        hda_audio_setup(st);

        hda_audio_set_amp(st);

        hda_audio_set_running(st, a->running_real[st->output * 16 + st->stream]);

    }

    return 0;

}
