static int v4l2_set_parameters(AVFormatContext *s1)

{

    struct video_data *s = s1->priv_data;

    struct v4l2_standard standard = { 0 };

    struct v4l2_streamparm streamparm = { 0 };

    struct v4l2_fract *tpf = &streamparm.parm.capture.timeperframe;

    AVRational framerate_q = { 0 };

    int i, ret;



    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;



    if (s->framerate &&

        (ret = av_parse_video_rate(&framerate_q, s->framerate)) < 0) {

        av_log(s1, AV_LOG_ERROR, "Could not parse framerate '%s'.\n",

               s->framerate);

        return ret;

    }



    if (s->standard) {

        av_log(s1, AV_LOG_DEBUG, "The V4L2 driver set standard: %s\n",

               s->standard);

        /* set tv standard */

        for(i=0;;i++) {

            standard.index = i;

            ret = v4l2_ioctl(s->fd, VIDIOC_ENUMSTD, &standard);

            if (ret < 0 || !av_strcasecmp(standard.name, s->standard))

                break;

        }

        if (ret < 0) {

            av_log(s1, AV_LOG_ERROR, "Unknown standard '%s'\n", s->standard);

            return ret;

        }



        av_log(s1, AV_LOG_DEBUG,

               "The V4L2 driver set standard: %s, id: %"PRIu64"\n",

               s->standard, (uint64_t)standard.id);

        if (v4l2_ioctl(s->fd, VIDIOC_S_STD, &standard.id) < 0) {

            av_log(s1, AV_LOG_ERROR,

                   "The V4L2 driver ioctl set standard(%s) failed\n",

                   s->standard);

            return AVERROR(EIO);

        }

    }



    if (framerate_q.num && framerate_q.den) {

        av_log(s1, AV_LOG_DEBUG, "Setting time per frame to %d/%d\n",

               framerate_q.den, framerate_q.num);

        tpf->numerator   = framerate_q.den;

        tpf->denominator = framerate_q.num;



        if (v4l2_ioctl(s->fd, VIDIOC_S_PARM, &streamparm) != 0) {

            av_log(s1, AV_LOG_ERROR,

                   "ioctl set time per frame(%d/%d) failed\n",

                   framerate_q.den, framerate_q.num);

            return AVERROR(EIO);

        }



        if (framerate_q.num != tpf->denominator ||

            framerate_q.den != tpf->numerator) {

            av_log(s1, AV_LOG_INFO,

                   "The driver changed the time per frame from "

                   "%d/%d to %d/%d\n",

                   framerate_q.den, framerate_q.num,

                   tpf->numerator, tpf->denominator);

        }

    } else {

        if (v4l2_ioctl(s->fd, VIDIOC_G_PARM, &streamparm) != 0) {

            av_log(s1, AV_LOG_ERROR, "ioctl(VIDIOC_G_PARM): %s\n",

                   strerror(errno));

            return AVERROR(errno);

        }

    }

    s1->streams[0]->avg_frame_rate.num = tpf->denominator;

    s1->streams[0]->avg_frame_rate.den = tpf->numerator;



    return 0;

}
