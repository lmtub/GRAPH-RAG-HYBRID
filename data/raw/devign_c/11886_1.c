static int v4l2_set_parameters(AVFormatContext *s1, AVFormatParameters *ap)

{

    struct video_data *s = s1->priv_data;

    struct v4l2_input input;

    struct v4l2_standard standard;

    struct v4l2_streamparm streamparm = { 0 };

    struct v4l2_fract *tpf = &streamparm.parm.capture.timeperframe;

    int i;



    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;



#if FF_API_FORMAT_PARAMETERS

    if (ap->channel > 0)

        s->channel = ap->channel;

#endif



    /* set tv video input */

    memset (&input, 0, sizeof (input));

    input.index = s->channel;

    if (ioctl(s->fd, VIDIOC_ENUMINPUT, &input) < 0) {

        av_log(s1, AV_LOG_ERROR, "The V4L2 driver ioctl enum input failed:\n");

        return AVERROR(EIO);

    }



    av_log(s1, AV_LOG_DEBUG, "The V4L2 driver set input_id: %d, input: %s\n",

            s->channel, input.name);

    if (ioctl(s->fd, VIDIOC_S_INPUT, &input.index) < 0) {

        av_log(s1, AV_LOG_ERROR, "The V4L2 driver ioctl set input(%d) failed\n",

                s->channel);

        return AVERROR(EIO);

    }



#if FF_API_FORMAT_PARAMETERS

    if (ap->standard) {

        av_freep(&s->standard);

        s->standard = av_strdup(ap->standard);

    }

#endif



    if (s->standard) {

        av_log(s1, AV_LOG_DEBUG, "The V4L2 driver set standard: %s\n",

               s->standard);

        /* set tv standard */

        memset (&standard, 0, sizeof (standard));

        for(i=0;;i++) {

            standard.index = i;

            if (ioctl(s->fd, VIDIOC_ENUMSTD, &standard) < 0) {

                av_log(s1, AV_LOG_ERROR, "The V4L2 driver ioctl set standard(%s) failed\n",

                       s->standard);

                return AVERROR(EIO);

            }



            if (!strcasecmp(standard.name, s->standard)) {

                break;

            }

        }



        av_log(s1, AV_LOG_DEBUG, "The V4L2 driver set standard: %s, id: %"PRIu64"\n",

               s->standard, (uint64_t)standard.id);

        if (ioctl(s->fd, VIDIOC_S_STD, &standard.id) < 0) {

            av_log(s1, AV_LOG_ERROR, "The V4L2 driver ioctl set standard(%s) failed\n",

                   s->standard);

            return AVERROR(EIO);

        }

    }

    av_freep(&s->standard);



    if (ap->time_base.num && ap->time_base.den) {

        av_log(s1, AV_LOG_DEBUG, "Setting time per frame to %d/%d\n",

               ap->time_base.num, ap->time_base.den);

        tpf->numerator = ap->time_base.num;

        tpf->denominator = ap->time_base.den;

        if (ioctl(s->fd, VIDIOC_S_PARM, &streamparm) != 0) {

            av_log(s1, AV_LOG_ERROR,

                   "ioctl set time per frame(%d/%d) failed\n",

                   ap->time_base.num, ap->time_base.den);

            return AVERROR(EIO);

        }



        if (ap->time_base.den != tpf->denominator ||

            ap->time_base.num != tpf->numerator) {

            av_log(s1, AV_LOG_INFO,

                   "The driver changed the time per frame from %d/%d to %d/%d\n",

                   ap->time_base.num, ap->time_base.den,

                   tpf->numerator, tpf->denominator);

        }

    } else {

        /* if timebase value is not set in ap, read the timebase value from the driver */

        if (ioctl(s->fd, VIDIOC_G_PARM, &streamparm) != 0) {

            av_log(s1, AV_LOG_ERROR, "ioctl(VIDIOC_G_PARM): %s\n", strerror(errno));

            return AVERROR(errno);

        }

    }

    ap->time_base.num = tpf->numerator;

    ap->time_base.den = tpf->denominator;



    return 0;

}
