static int oss_ctl_in (HWVoiceIn *hw, int cmd, ...)

{

    OSSVoiceIn *oss = (OSSVoiceIn *) hw;



    switch (cmd) {

    case VOICE_ENABLE:

        {

            va_list ap;

            int poll_mode;



            va_start (ap, cmd);

            poll_mode = va_arg (ap, int);

            va_end (ap);



            if (poll_mode && oss_poll_in (hw)) {

                poll_mode = 0;

            }

            hw->poll_mode = poll_mode;

        }

        break;



    case VOICE_DISABLE:

        if (hw->poll_mode) {

            hw->poll_mode = 0;

            qemu_set_fd_handler (oss->fd, NULL, NULL, NULL);

        }

        break;

    }

    return 0;

}
