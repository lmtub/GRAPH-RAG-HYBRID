static int oss_init_out (HWVoiceOut *hw, struct audsettings *as)

{

    OSSVoiceOut *oss = (OSSVoiceOut *) hw;

    struct oss_params req, obt;

    int endianness;

    int err;

    int fd;

    audfmt_e effective_fmt;

    struct audsettings obt_as;



    oss->fd = -1;



    req.fmt = aud_to_ossfmt (as->fmt, as->endianness);

    req.freq = as->freq;

    req.nchannels = as->nchannels;

    req.fragsize = conf.fragsize;

    req.nfrags = conf.nfrags;



    if (oss_open (0, &req, &obt, &fd)) {

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

        dolog ("warning: Misaligned DAC buffer, size %d, alignment %d\n",

               obt.nfrags * obt.fragsize, hw->info.align + 1);

    }



    hw->samples = (obt.nfrags * obt.fragsize) >> hw->info.shift;



    oss->mmapped = 0;

    if (conf.try_mmap) {

        oss->pcm_buf = mmap (

            NULL,

            hw->samples << hw->info.shift,

            PROT_READ | PROT_WRITE,

            MAP_SHARED,

            fd,

            0

            );

        if (oss->pcm_buf == MAP_FAILED) {

            oss_logerr (errno, "Failed to map %d bytes of DAC\n",

                        hw->samples << hw->info.shift);

        }

        else {

            int err;

            int trig = 0;

            if (ioctl (fd, SNDCTL_DSP_SETTRIGGER, &trig) < 0) {

                oss_logerr (errno, "SNDCTL_DSP_SETTRIGGER 0 failed\n");

            }

            else {

                trig = PCM_ENABLE_OUTPUT;

                if (ioctl (fd, SNDCTL_DSP_SETTRIGGER, &trig) < 0) {

                    oss_logerr (

                        errno,

                        "SNDCTL_DSP_SETTRIGGER PCM_ENABLE_OUTPUT failed\n"

                        );

                }

                else {

                    oss->mmapped = 1;

                }

            }



            if (!oss->mmapped) {

                err = munmap (oss->pcm_buf, hw->samples << hw->info.shift);

                if (err) {

                    oss_logerr (errno, "Failed to unmap buffer %p size %d\n",

                                oss->pcm_buf, hw->samples << hw->info.shift);

                }

            }

        }

    }



    if (!oss->mmapped) {

        oss->pcm_buf = audio_calloc (

            AUDIO_FUNC,

            hw->samples,

            1 << hw->info.shift

            );

        if (!oss->pcm_buf) {

            dolog (

                "Could not allocate DAC buffer (%d samples, each %d bytes)\n",

                hw->samples,

                1 << hw->info.shift

                );

            oss_anal_close (&fd);

            return -1;

        }

    }



    oss->fd = fd;

    return 0;

}
