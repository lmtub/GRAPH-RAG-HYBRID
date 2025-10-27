static int qesd_init_in (HWVoiceIn *hw, audsettings_t *as)

{

    ESDVoiceIn *esd = (ESDVoiceIn *) hw;

    audsettings_t obt_as = *as;

    int esdfmt = ESD_STREAM | ESD_RECORD;

    int err;

    sigset_t set, old_set;



    sigfillset (&set);



    esdfmt |= (as->nchannels == 2) ? ESD_STEREO : ESD_MONO;

    switch (as->fmt) {

    case AUD_FMT_S8:

    case AUD_FMT_U8:

        esdfmt |= ESD_BITS8;

        obt_as.fmt = AUD_FMT_U8;

        break;



    case AUD_FMT_S16:

    case AUD_FMT_U16:

        esdfmt |= ESD_BITS16;

        obt_as.fmt = AUD_FMT_S16;

        break;



    case AUD_FMT_S32:

    case AUD_FMT_U32:

        dolog ("Will use 16 instead of 32 bit samples\n");

        esdfmt |= ESD_BITS16;

        obt_as.fmt = AUD_FMT_S16;

        break;

    }

    obt_as.endianness = AUDIO_HOST_ENDIANNESS;



    audio_pcm_init_info (&hw->info, &obt_as);



    hw->samples = conf.samples;

    esd->pcm_buf = audio_calloc (AUDIO_FUNC, hw->samples, 1 << hw->info.shift);

    if (!esd->pcm_buf) {

        dolog ("Could not allocate buffer (%d bytes)\n",

               hw->samples << hw->info.shift);

        return -1;

    }



    esd->fd = -1;



    err = pthread_sigmask (SIG_BLOCK, &set, &old_set);

    if (err) {

        qesd_logerr (err, "pthread_sigmask failed\n");

        goto fail1;

    }



    esd->fd = esd_record_stream (esdfmt, as->freq, conf.adc_host, NULL);

    if (esd->fd < 0) {

        qesd_logerr (errno, "esd_record_stream failed\n");

        goto fail2;

    }



    if (audio_pt_init (&esd->pt, qesd_thread_in, esd, AUDIO_CAP, AUDIO_FUNC)) {

        goto fail3;

    }



    err = pthread_sigmask (SIG_SETMASK, &old_set, NULL);

    if (err) {

        qesd_logerr (err, "pthread_sigmask(restore) failed\n");

    }



    return 0;



 fail3:

    if (close (esd->fd)) {

        qesd_logerr (errno, "%s: close on esd socket(%d) failed\n",

                     AUDIO_FUNC, esd->fd);

    }

    esd->fd = -1;



 fail2:

    err = pthread_sigmask (SIG_SETMASK, &old_set, NULL);

    if (err) {

        qesd_logerr (err, "pthread_sigmask(restore) failed\n");

    }



 fail1:

    qemu_free (esd->pcm_buf);

    esd->pcm_buf = NULL;

    return -1;

}
