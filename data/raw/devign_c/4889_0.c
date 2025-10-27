static int fmod_init_in (HWVoiceIn *hw, audsettings_t *as)

{

    int bits16, mode;

    FMODVoiceIn *fmd = (FMODVoiceIn *) hw;

    audsettings_t obt_as = *as;



    if (conf.broken_adc) {

        return -1;

    }



    mode = aud_to_fmodfmt (as->fmt, as->nchannels == 2 ? 1 : 0);

    fmd->fmod_sample = FSOUND_Sample_Alloc (

        FSOUND_FREE,            /* index */

        conf.nb_samples,        /* length */

        mode,                   /* mode */

        as->freq,               /* freq */

        255,                    /* volume */

        128,                    /* pan */

        255                     /* priority */

        );



    if (!fmd->fmod_sample) {

        fmod_logerr2 ("ADC", "Failed to allocate FMOD sample\n");

        return -1;

    }



    /* FMOD always operates on little endian frames? */

    obt_as.endianness = 0;

    audio_pcm_init_info (&hw->info, &obt_as);

    bits16 = (mode & FSOUND_16BITS) != 0;

    hw->samples = conf.nb_samples;

    return 0;

}
