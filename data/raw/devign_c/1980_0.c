static int alsa_run_in (HWVoiceIn *hw)

{

    ALSAVoiceIn *alsa = (ALSAVoiceIn *) hw;

    int hwshift = hw->info.shift;

    int i;

    int live = audio_pcm_hw_get_live_in (hw);

    int dead = hw->samples - live;

    int decr;

    struct {

        int add;

        int len;

    } bufs[2] = {

        { hw->wpos, 0 },

        { 0, 0 }

    };

    snd_pcm_sframes_t avail;

    snd_pcm_uframes_t read_samples = 0;



    if (!dead) {

        return 0;

    }



    avail = alsa_get_avail (alsa->handle);

    if (avail < 0) {

        dolog ("Could not get number of captured frames\n");

        return 0;

    }



    if (!avail && (snd_pcm_state (alsa->handle) == SND_PCM_STATE_PREPARED)) {

        avail = hw->samples;

    }



    decr = audio_MIN (dead, avail);

    if (!decr) {

        return 0;

    }



    if (hw->wpos + decr > hw->samples) {

        bufs[0].len = (hw->samples - hw->wpos);

        bufs[1].len = (decr - (hw->samples - hw->wpos));

    }

    else {

        bufs[0].len = decr;

    }



    for (i = 0; i < 2; ++i) {

        void *src;

        st_sample_t *dst;

        snd_pcm_sframes_t nread;

        snd_pcm_uframes_t len;



        len = bufs[i].len;



        src = advance (alsa->pcm_buf, bufs[i].add << hwshift);

        dst = hw->conv_buf + bufs[i].add;



        while (len) {

            nread = snd_pcm_readi (alsa->handle, src, len);



            if (nread <= 0) {

                switch (nread) {

                case 0:

                    if (conf.verbose) {

                        dolog ("Failed to read %ld frames (read zero)\n", len);

                    }

                    goto exit;



                case -EPIPE:

                    if (alsa_recover (alsa->handle)) {

                        alsa_logerr (nread, "Failed to read %ld frames\n", len);

                        goto exit;

                    }

                    if (conf.verbose) {

                        dolog ("Recovering from capture xrun\n");

                    }

                    continue;



                case -EAGAIN:

                    goto exit;



                default:

                    alsa_logerr (

                        nread,

                        "Failed to read %ld frames from %p\n",

                        len,

                        src

                        );

                    goto exit;

                }

            }



            hw->conv (dst, src, nread, &nominal_volume);



            src = advance (src, nread << hwshift);

            dst += nread;



            read_samples += nread;

            len -= nread;

        }

    }



 exit:

    hw->wpos = (hw->wpos + read_samples) % hw->samples;

    return read_samples;

}
