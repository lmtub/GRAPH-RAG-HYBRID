static int oss_init_in (HWVoiceIn *hw, audsettings_t *as)

{

    OSSVoiceIn *oss = (OSSVoiceIn *) hw;

    struct oss_params req, obt;

    int endianness;

    int err;

    int fd;

    audfmt_e effective_fmt;

    audsettings_t obt_as;



    oss->fd = -1;



    req.fmt = aud_to_ossfmt (as->fmt);

    req.freq = as->freq;

    req.nchannels = as->nchannels;

    req.fragsize = conf.fragsize;

    req.nfrags = conf.nfrags;

    if (oss_open (1, &req, &obt, &fd)) {

        return -1;

    }



    err = oss_to_audfmt (obt.fmt, &effective_fmt, &endianness);

    if (err) {

        oss_anal_close (&fd);

        return -1;

    }



    obt_as.freq = obt.freq;

    obt_as.nchannels = obt.nchannels;

    obt_as.fmt = effective_fmt;

    obt_as.endianness = endianness;



    audio_pcm_init_info (&hw->info, &obt_as);

    oss->nfrags = obt.nfrags;

    oss->fragsize = obt.fragsize;



    if (obt.nfrags * obt.fragsize & hw->info.align) {

        dolog ("warning: Misaligned ADC buffer, size %d, alignment %d\n",

               obt.nfrags * obt.fragsize, hw->info.align + 1);

    }



    hw->samples = (obt.nfrags * obt.fragsize) >> hw->info.shift;

    oss->pcm_buf = audio_calloc (AUDIO_FUNC, hw->samples, 1 << hw->info.shift);

    if (!oss->pcm_buf) {

        dolog ("Could not allocate ADC buffer (%d samples, each %d bytes)\n",

               hw->samples, 1 << hw->info.shift);

        oss_anal_close (&fd);

        return -1;

    }



    oss->fd = fd;

    return 0;

}
