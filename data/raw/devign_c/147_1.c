static int alsa_init_in (HWVoiceIn *hw, struct audsettings *as)

{

    ALSAVoiceIn *alsa = (ALSAVoiceIn *) hw;

    struct alsa_params_req req;

    struct alsa_params_obt obt;

    snd_pcm_t *handle;

    struct audsettings obt_as;



    req.fmt = aud_to_alsafmt (as->fmt, as->endianness);

    req.freq = as->freq;

    req.nchannels = as->nchannels;

    req.period_size = conf.period_size_in;

    req.buffer_size = conf.buffer_size_in;

    req.size_in_usec = conf.size_in_usec_in;

    req.override_mask =

        (conf.period_size_in_overridden ? 1 : 0) |

        (conf.buffer_size_in_overridden ? 2 : 0);



    if (alsa_open (1, &req, &obt, &handle)) {

        return -1;

    }



    obt_as.freq = obt.freq;

    obt_as.nchannels = obt.nchannels;

    obt_as.fmt = obt.fmt;

    obt_as.endianness = obt.endianness;



    audio_pcm_init_info (&hw->info, &obt_as);

    hw->samples = obt.samples;



    alsa->pcm_buf = audio_calloc (AUDIO_FUNC, hw->samples, 1 << hw->info.shift);

    if (!alsa->pcm_buf) {

        dolog ("Could not allocate ADC buffer (%d samples, each %d bytes)\n",

               hw->samples, 1 << hw->info.shift);

        alsa_anal_close1 (&handle);

        return -1;

    }



    alsa->handle = handle;

    return 0;

}
