static int fmod_init_out (HWVoiceOut *hw, audsettings_t *as)

{

    int bits16, mode, channel;

    FMODVoiceOut *fmd = (FMODVoiceOut *) hw;

    audsettings_t obt_as = *as;



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

        fmod_logerr2 ("DAC", "Failed to allocate FMOD sample\n");

        return -1;

    }



    channel = FSOUND_PlaySoundEx (FSOUND_FREE, fmd->fmod_sample, 0, 1);

    if (channel < 0) {

        fmod_logerr2 ("DAC", "Failed to start playing sound\n");

        FSOUND_Sample_Free (fmd->fmod_sample);

        return -1;

    }

    fmd->channel = channel;



    /* FMOD always operates on little endian frames? */

    obt_as.endianness = 0;

    audio_pcm_init_info (&hw->info, &obt_as);

    bits16 = (mode & FSOUND_16BITS) != 0;

    hw->samples = conf.nb_samples;

    return 0;

}
