static int alsa_run_out (HWVoiceOut *hw)

{

    ALSAVoiceOut *alsa = (ALSAVoiceOut *) hw;

    int rpos, live, decr;

    int samples;

    uint8_t *dst;

    st_sample_t *src;

    snd_pcm_sframes_t avail;



    live = audio_pcm_hw_get_live_out (hw);

    if (!live) {

        return 0;

    }



    avail = alsa_get_avail (alsa->handle);

    if (avail < 0) {

        dolog ("Could not get number of available playback frames\n");

        return 0;

    }



    decr = audio_MIN (live, avail);

    samples = decr;

    rpos = hw->rpos;

    while (samples) {

        int left_till_end_samples = hw->samples - rpos;

        int len = audio_MIN (samples, left_till_end_samples);

        snd_pcm_sframes_t written;



        src = hw->mix_buf + rpos;

        dst = advance (alsa->pcm_buf, rpos << hw->info.shift);



        hw->clip (dst, src, len);



        while (len) {

            written = snd_pcm_writei (alsa->handle, dst, len);



            if (written <= 0) {

                switch (written) {

                case 0:

                    if (conf.verbose) {

                        dolog ("Failed to write %d frames (wrote zero)\n", len);

                    }

                    goto exit;



                case -EPIPE:

                    if (alsa_recover (alsa->handle)) {

                        alsa_logerr (written, "Failed to write %d frames\n",

                                     len);

                        goto exit;

                    }

                    if (conf.verbose) {

                        dolog ("Recovering from playback xrun\n");

                    }

                    continue;



                case -EAGAIN:

                    goto exit;



                default:

                    alsa_logerr (written, "Failed to write %d frames to %p\n",

                                 len, dst);

                    goto exit;

                }

            }



            rpos = (rpos + written) % hw->samples;

            samples -= written;

            len -= written;

            dst = advance (dst, written << hw->info.shift);

            src += written;

        }

    }



 exit:

    hw->rpos = rpos;

    return decr;

}
