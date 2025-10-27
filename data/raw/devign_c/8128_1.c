static int v4l2_read_header(AVFormatContext *s1, AVFormatParameters *ap)

{

    struct video_data *s = s1->priv_data;

    AVStream *st;

    int width, height;

    int res, frame_rate, frame_rate_base;

    uint32_t desired_format, capabilities;



    if (ap->width <= 0 || ap->height <= 0 || ap->time_base.den <= 0) {

        av_log(s1, AV_LOG_ERROR, "Missing/Wrong width, height or framerate\n");



        return -1;

    }



    width = ap->width;

    height = ap->height;

    frame_rate = ap->time_base.den;

    frame_rate_base = ap->time_base.num;



    if((unsigned)width > 32767 || (unsigned)height > 32767) {

        av_log(s1, AV_LOG_ERROR, "Wrong size %dx%d\n", width, height);



        return -1;

    }



    st = av_new_stream(s1, 0);

    if (!st) {

        return AVERROR(ENOMEM);

    }

    av_set_pts_info(st, 64, 1, 1000000); /* 64 bits pts in us */



    s->width = width;

    s->height = height;

    s->frame_rate      = frame_rate;

    s->frame_rate_base = frame_rate_base;



    capabilities = 0;

    s->fd = device_open(s1, &capabilities);

    if (s->fd < 0) {

        av_free(st);



        return AVERROR(EIO);

    }

    av_log(s1, AV_LOG_INFO, "[%d]Capabilities: %x\n", s->fd, capabilities);



    desired_format = fmt_ff2v4l(ap->pix_fmt);

    if (desired_format == 0 || (device_init(s1, &width, &height, desired_format) < 0)) {

        int i, done;



        done = 0; i = 0;

        while (!done) {

            desired_format = fmt_conversion_table[i].v4l2_fmt;

            if (device_init(s1, &width, &height, desired_format) < 0) {

                desired_format = 0;

                i++;

            } else {

               done = 1;

            }

            if (i == sizeof(fmt_conversion_table) / sizeof(struct fmt_map)) {

               done = 1;

            }

        }

    }

    if (desired_format == 0) {

        av_log(s1, AV_LOG_ERROR, "Cannot find a proper format.\n");

        close(s->fd);

        av_free(st);



        return AVERROR(EIO);

    }

    s->frame_format = desired_format;



    if( v4l2_set_parameters( s1, ap ) < 0 )

        return AVERROR(EIO);



    st->codec->pix_fmt = fmt_v4l2ff(desired_format);

    s->frame_size = avpicture_get_size(st->codec->pix_fmt, width, height);

    if (capabilities & V4L2_CAP_STREAMING) {

        s->io_method = io_mmap;

        res = mmap_init(s1);

        if (res == 0) {

            res = mmap_start(s1);

        }

    } else {

        s->io_method = io_read;

        res = read_init(s1);

    }

    if (res < 0) {

        close(s->fd);

        av_free(st);



        return AVERROR(EIO);

    }

    s->top_field_first = first_field(s->fd);



    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_RAWVIDEO;

    st->codec->width = width;

    st->codec->height = height;

    st->codec->time_base.den = frame_rate;

    st->codec->time_base.num = frame_rate_base;

    st->codec->bit_rate = s->frame_size * 1/av_q2d(st->codec->time_base) * 8;



    return 0;

}
