static int qpa_init_out (HWVoiceOut *hw, struct audsettings *as)

{

    int error;

    static pa_sample_spec ss;

    static pa_buffer_attr ba;

    struct audsettings obt_as = *as;

    PAVoiceOut *pa = (PAVoiceOut *) hw;



    ss.format = audfmt_to_pa (as->fmt, as->endianness);

    ss.channels = as->nchannels;

    ss.rate = as->freq;



    /*

     * qemu audio tick runs at 250 Hz (by default), so processing

     * data chunks worth 4 ms of sound should be a good fit.

     */

    ba.tlength = pa_usec_to_bytes (4 * 1000, &ss);

    ba.minreq = pa_usec_to_bytes (2 * 1000, &ss);

    ba.maxlength = -1;

    ba.prebuf = -1;



    obt_as.fmt = pa_to_audfmt (ss.format, &obt_as.endianness);



    pa->s = pa_simple_new (

        conf.server,

        "qemu",

        PA_STREAM_PLAYBACK,

        conf.sink,

        "pcm.playback",

        &ss,

        NULL,                   /* channel map */

        &ba,                    /* buffering attributes */

        &error

        );

    if (!pa->s) {

        qpa_logerr (error, "pa_simple_new for playback failed\n");

        goto fail1;

    }



    audio_pcm_init_info (&hw->info, &obt_as);

    hw->samples = conf.samples;

    pa->pcm_buf = audio_calloc (AUDIO_FUNC, hw->samples, 1 << hw->info.shift);

    pa->rpos = hw->rpos;

    if (!pa->pcm_buf) {

        dolog ("Could not allocate buffer (%d bytes)\n",

               hw->samples << hw->info.shift);

        goto fail2;

    }



    if (audio_pt_init (&pa->pt, qpa_thread_out, hw, AUDIO_CAP, AUDIO_FUNC)) {

        goto fail3;

    }



    return 0;



 fail3:

    g_free (pa->pcm_buf);

    pa->pcm_buf = NULL;

 fail2:

    pa_simple_free (pa->s);

    pa->s = NULL;

 fail1:

    return -1;

}
