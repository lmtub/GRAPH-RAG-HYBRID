static int v4l2_set_parameters(AVFormatContext *s1)

{

    struct video_data *s = s1->priv_data;

    struct v4l2_standard standard = { 0 };

    struct v4l2_streamparm streamparm = { 0 };

    struct v4l2_fract *tpf;

    AVRational framerate_q = { 0 };

    int i, ret;



    if (s->framerate &&

        (ret = av_parse_video_rate(&framerate_q, s->framerate)) < 0) {

        av_log(s1, AV_LOG_ERROR, "Could not parse framerate '%s'.\n",

               s->framerate);

        return ret;

    }



    if (s->standard) {

        if (s->std_id) {

            ret = 0;

            av_log(s1, AV_LOG_DEBUG, "Setting standard: %s\n", s->standard);

            /* set tv standard */

            for (i = 0; ; i++) {

                standard.index = i;

                if (v4l2_ioctl(s->fd, VIDIOC_ENUMSTD, &standard) < 0) {

                    ret = AVERROR(errno);

                    break;

                }

                if (!av_strcasecmp(standard.name, s->standard))

                    break;

            }

            if (ret < 0) {

                av_log(s1, AV_LOG_ERROR, "Unknown or unsupported standard '%s'\n", s->standard);

                return ret;

            }



            if (v4l2_ioctl(s->fd, VIDIOC_S_STD, &standard.id) < 0) {

                ret = AVERROR(errno);

                av_log(s1, AV_LOG_ERROR, "ioctl(VIDIOC_S_STD): %s\n", av_err2str(ret));

                return ret;

            }

        } else {

            av_log(s1, AV_LOG_WARNING,

                   "This device does not support any standard\n");

        }

    }



    /* get standard */

    if (v4l2_ioctl(s->fd, VIDIOC_G_STD, &s->std_id) == 0) {

        tpf = &standard.frameperiod;

        for (i = 0; ; i++) {

            standard.index = i;

            if (v4l2_ioctl(s->fd, VIDIOC_ENUMSTD, &standard) < 0) {

                ret = AVERROR(errno);

                if (ret == AVERROR(EINVAL)) {

                    tpf = &streamparm.parm.capture.timeperframe;

                    break;

                }

                av_log(s1, AV_LOG_ERROR, "ioctl(VIDIOC_ENUMSTD): %s\n", av_err2str(ret));

                return ret;

            }

            if (standard.id == s->std_id) {

                av_log(s1, AV_LOG_DEBUG,

                       "Current standard: %s, id: %"PRIx64", frameperiod: %d/%d\n",

                       standard.name, (uint64_t)standard.id, tpf->numerator, tpf->denominator);

                break;

            }

        }

    } else {

        tpf = &streamparm.parm.capture.timeperframe;

    }



    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (v4l2_ioctl(s->fd, VIDIOC_G_PARM, &streamparm) < 0) {

        ret = AVERROR(errno);

        av_log(s1, AV_LOG_ERROR, "ioctl(VIDIOC_G_PARM): %s\n", av_err2str(ret));

        return ret;

    }



    if (framerate_q.num && framerate_q.den) {

        if (streamparm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) {

            tpf = &streamparm.parm.capture.timeperframe;



            av_log(s1, AV_LOG_DEBUG, "Setting time per frame to %d/%d\n",

                   framerate_q.den, framerate_q.num);

            tpf->numerator   = framerate_q.den;

            tpf->denominator = framerate_q.num;



            if (v4l2_ioctl(s->fd, VIDIOC_S_PARM, &streamparm) < 0) {

                ret = AVERROR(errno);

                av_log(s1, AV_LOG_ERROR, "ioctl(VIDIOC_S_PARM): %s\n", av_err2str(ret));

                return ret;

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

            av_log(s1, AV_LOG_WARNING,

                   "The driver does not allow to change time per frame\n");

        }

    }

    if (tpf->denominator > 0 && tpf->numerator > 0) {

        s1->streams[0]->avg_frame_rate.num = tpf->denominator;

        s1->streams[0]->avg_frame_rate.den = tpf->numerator;

        s1->streams[0]->r_frame_rate = s1->streams[0]->avg_frame_rate;

    } else

        av_log(s1, AV_LOG_WARNING, "Time per frame unknown\n");



    return 0;

}
