static int oss_ctl_out (HWVoiceOut *hw, int cmd, ...)

{

    int trig;

    OSSVoiceOut *oss = (OSSVoiceOut *) hw;



    switch (cmd) {

    case VOICE_ENABLE:

        {

            va_list ap;

            int poll_mode;



            va_start (ap, cmd);

            poll_mode = va_arg (ap, int);

            va_end (ap);



            ldebug ("enabling voice\n");

            if (poll_mode && oss_poll_out (hw)) {

                poll_mode = 0;

            }

            hw->poll_mode = poll_mode;



            if (!oss->mmapped) {

                return 0;

            }



            audio_pcm_info_clear_buf (&hw->info, oss->pcm_buf, hw->samples);

            trig = PCM_ENABLE_OUTPUT;

            if (ioctl (oss->fd, SNDCTL_DSP_SETTRIGGER, &trig) < 0) {

                oss_logerr (

                    errno,

                    "SNDCTL_DSP_SETTRIGGER PCM_ENABLE_OUTPUT failed\n"

                    );

                return -1;

            }

        }

        break;



    case VOICE_DISABLE:

        if (hw->poll_mode) {

            qemu_set_fd_handler (oss->fd, NULL, NULL, NULL);

            hw->poll_mode = 0;

        }



        if (!oss->mmapped) {

            return 0;

        }



        ldebug ("disabling voice\n");

        trig = 0;

        if (ioctl (oss->fd, SNDCTL_DSP_SETTRIGGER, &trig) < 0) {

            oss_logerr (errno, "SNDCTL_DSP_SETTRIGGER 0 failed\n");

            return -1;

        }

        break;

    }

    return 0;

}
