int ff_oss_audio_open(AVFormatContext *s1, int is_output,

                      const char *audio_device)

{

    OSSAudioData *s = s1->priv_data;

    int audio_fd;

    int tmp, err;

    char *flip = getenv("AUDIO_FLIP_LEFT");



    if (is_output)

        audio_fd = avpriv_open(audio_device, O_WRONLY);

    else

        audio_fd = avpriv_open(audio_device, O_RDONLY);

    if (audio_fd < 0) {

        av_log(s1, AV_LOG_ERROR, "%s: %s\n", audio_device, strerror(errno));

        return AVERROR(EIO);

    }



    if (flip && *flip == '1') {

        s->flip_left = 1;

    }



    /* non blocking mode */

    if (!is_output)

        fcntl(audio_fd, F_SETFL, O_NONBLOCK);



    s->frame_size = OSS_AUDIO_BLOCK_SIZE;



    /* select format : favour native format */

    err = ioctl(audio_fd, SNDCTL_DSP_GETFMTS, &tmp);



#if HAVE_BIGENDIAN

    if (tmp & AFMT_S16_BE) {

        tmp = AFMT_S16_BE;

    } else if (tmp & AFMT_S16_LE) {

        tmp = AFMT_S16_LE;

    } else {

        tmp = 0;

    }

#else

    if (tmp & AFMT_S16_LE) {

        tmp = AFMT_S16_LE;

    } else if (tmp & AFMT_S16_BE) {

        tmp = AFMT_S16_BE;

    } else {

        tmp = 0;

    }

#endif



    switch(tmp) {

    case AFMT_S16_LE:

        s->codec_id = AV_CODEC_ID_PCM_S16LE;

        break;

    case AFMT_S16_BE:

        s->codec_id = AV_CODEC_ID_PCM_S16BE;

        break;

    default:

        av_log(s1, AV_LOG_ERROR, "Soundcard does not support 16 bit sample format\n");

        close(audio_fd);

        return AVERROR(EIO);

    }

    err=ioctl(audio_fd, SNDCTL_DSP_SETFMT, &tmp);

    if (err < 0) {

        av_log(s1, AV_LOG_ERROR, "SNDCTL_DSP_SETFMT: %s\n", strerror(errno));

        goto fail;

    }



    tmp = (s->channels == 2);

    err = ioctl(audio_fd, SNDCTL_DSP_STEREO, &tmp);

    if (err < 0) {

        av_log(s1, AV_LOG_ERROR, "SNDCTL_DSP_STEREO: %s\n", strerror(errno));

        goto fail;

    }



    tmp = s->sample_rate;

    err = ioctl(audio_fd, SNDCTL_DSP_SPEED, &tmp);

    if (err < 0) {

        av_log(s1, AV_LOG_ERROR, "SNDCTL_DSP_SPEED: %s\n", strerror(errno));

        goto fail;

    }

    s->sample_rate = tmp; /* store real sample rate */

    s->fd = audio_fd;



    return 0;

 fail:

    close(audio_fd);

    return AVERROR(EIO);

}
