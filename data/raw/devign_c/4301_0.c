static int oss_run_in (HWVoiceIn *hw)

{

    OSSVoiceIn *oss = (OSSVoiceIn *) hw;

    int hwshift = hw->info.shift;

    int i;

    int live = audio_pcm_hw_get_live_in (hw);

    int dead = hw->samples - live;

    size_t read_samples = 0;

    struct {

        int add;

        int len;

    } bufs[2] = {

        { hw->wpos, 0 },

        { 0, 0 }

    };



    if (!dead) {

        return 0;

    }



    if (hw->wpos + dead > hw->samples) {

        bufs[0].len = (hw->samples - hw->wpos) << hwshift;

        bufs[1].len = (dead - (hw->samples - hw->wpos)) << hwshift;

    }

    else {

        bufs[0].len = dead << hwshift;

    }





    for (i = 0; i < 2; ++i) {

        ssize_t nread;



        if (bufs[i].len) {

            void *p = advance (oss->pcm_buf, bufs[i].add << hwshift);

            nread = read (oss->fd, p, bufs[i].len);



            if (nread > 0) {

                if (nread & hw->info.align) {

                    dolog ("warning: Misaligned read %zd (requested %d), "

                           "alignment %d\n", nread, bufs[i].add << hwshift,

                           hw->info.align + 1);

                }

                read_samples += nread >> hwshift;

                hw->conv (hw->conv_buf + bufs[i].add, p, nread >> hwshift,

                          &nominal_volume);

            }



            if (bufs[i].len - nread) {

                if (nread == -1) {

                    switch (errno) {

                    case EINTR:

                    case EAGAIN:

                        break;

                    default:

                        oss_logerr (

                            errno,

                            "Failed to read %d bytes of audio (to %p)\n",

                            bufs[i].len, p

                            );

                        break;

                    }

                }

                break;

            }

        }

    }



    hw->wpos = (hw->wpos + read_samples) % hw->samples;

    return read_samples;

}
