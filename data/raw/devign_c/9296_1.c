static int grab_read_header(AVFormatContext *s1, AVFormatParameters *ap)

{

    VideoData *s = s1->priv_data;

    AVStream *st;

    int width, height;

    int video_fd, frame_size;

    int ret, frame_rate, frame_rate_base;

    int desired_palette, desired_depth;

    struct video_tuner tuner;

    struct video_audio audio;

    struct video_picture pict;

    int j;

    int vformat_num = sizeof(video_formats) / sizeof(video_formats[0]);



    if (ap->width <= 0 || ap->height <= 0 || ap->time_base.den <= 0) {

        av_log(s1, AV_LOG_ERROR, "Bad capture size (%dx%d) or wrong time base (%d)\n",

            ap->width, ap->height, ap->time_base.den);



        return -1;

    }



    width = ap->width;

    height = ap->height;

    frame_rate      = ap->time_base.den;

    frame_rate_base = ap->time_base.num;



    if((unsigned)width > 32767 || (unsigned)height > 32767) {

        av_log(s1, AV_LOG_ERROR, "Capture size is out of range: %dx%d\n",

            width, height);



        return -1;

    }



    st = av_new_stream(s1, 0);

    if (!st)

        return AVERROR(ENOMEM);

    av_set_pts_info(st, 64, 1, 1000000); /* 64 bits pts in us */



    s->width = width;

    s->height = height;

    s->frame_rate      = frame_rate;

    s->frame_rate_base = frame_rate_base;



    video_fd = open(s1->filename, O_RDWR);

    if (video_fd < 0) {

        av_log(s1, AV_LOG_ERROR, "%s: %s\n", s1->filename, strerror(errno));

        goto fail;

    }



    if (ioctl(video_fd,VIDIOCGCAP, &s->video_cap) < 0) {

        av_log(s1, AV_LOG_ERROR, "VIDIOCGCAP: %s\n", strerror(errno));

        goto fail;

    }



    if (!(s->video_cap.type & VID_TYPE_CAPTURE)) {

        av_log(s1, AV_LOG_ERROR, "Fatal: grab device does not handle capture\n");

        goto fail;

    }



    desired_palette = -1;

    desired_depth = -1;

    for (j = 0; j < vformat_num; j++) {

        if (ap->pix_fmt == video_formats[j].pix_fmt) {

            desired_palette = video_formats[j].palette;

            desired_depth = video_formats[j].depth;

            break;

        }

    }



    /* set tv standard */

    if (ap->standard && !ioctl(video_fd, VIDIOCGTUNER, &tuner)) {

        if (!strcasecmp(ap->standard, "pal"))

            tuner.mode = VIDEO_MODE_PAL;

        else if (!strcasecmp(ap->standard, "secam"))

            tuner.mode = VIDEO_MODE_SECAM;

        else

            tuner.mode = VIDEO_MODE_NTSC;

        ioctl(video_fd, VIDIOCSTUNER, &tuner);

    }



    /* unmute audio */

    audio.audio = 0;

    ioctl(video_fd, VIDIOCGAUDIO, &audio);

    memcpy(&s->audio_saved, &audio, sizeof(audio));

    audio.flags &= ~VIDEO_AUDIO_MUTE;

    ioctl(video_fd, VIDIOCSAUDIO, &audio);



    ioctl(video_fd, VIDIOCGPICT, &pict);

#if 0

    printf("v4l: colour=%d hue=%d brightness=%d constrast=%d whiteness=%d\n",

           pict.colour,

           pict.hue,

           pict.brightness,

           pict.contrast,

           pict.whiteness);

#endif

    /* try to choose a suitable video format */

    pict.palette = desired_palette;

    pict.depth= desired_depth;

    if (desired_palette == -1 || (ret = ioctl(video_fd, VIDIOCSPICT, &pict)) < 0) {

        for (j = 0; j < vformat_num; j++) {

            pict.palette = video_formats[j].palette;

            pict.depth = video_formats[j].depth;

            if (-1 != ioctl(video_fd, VIDIOCSPICT, &pict))

                break;

        }

        if (j >= vformat_num)

            goto fail1;

    }



    ret = ioctl(video_fd,VIDIOCGMBUF,&s->gb_buffers);

    if (ret < 0) {

        /* try to use read based access */

        struct video_window win;

        int val;



        win.x = 0;

        win.y = 0;

        win.width = width;

        win.height = height;

        win.chromakey = -1;

        win.flags = 0;



        ioctl(video_fd, VIDIOCSWIN, &win);



        s->frame_format = pict.palette;



        val = 1;

        ioctl(video_fd, VIDIOCCAPTURE, &val);



        s->time_frame = av_gettime() * s->frame_rate / s->frame_rate_base;

        s->use_mmap = 0;

    } else {

        s->video_buf = mmap(0,s->gb_buffers.size,PROT_READ|PROT_WRITE,MAP_SHARED,video_fd,0);

        if ((unsigned char*)-1 == s->video_buf) {

            s->video_buf = mmap(0,s->gb_buffers.size,PROT_READ|PROT_WRITE,MAP_PRIVATE,video_fd,0);

            if ((unsigned char*)-1 == s->video_buf) {

                av_log(s1, AV_LOG_ERROR, "mmap: %s\n", strerror(errno));

                goto fail;

            }

        }

        s->gb_frame = 0;

        s->time_frame = av_gettime() * s->frame_rate / s->frame_rate_base;



        /* start to grab the first frame */

        s->gb_buf.frame = s->gb_frame % s->gb_buffers.frames;

        s->gb_buf.height = height;

        s->gb_buf.width = width;

        s->gb_buf.format = pict.palette;



        ret = ioctl(video_fd, VIDIOCMCAPTURE, &s->gb_buf);

        if (ret < 0) {

            if (errno != EAGAIN) {

            fail1:

                av_log(s1, AV_LOG_ERROR, "Fatal: grab device does not support suitable format\n");

            } else {

                av_log(s1, AV_LOG_ERROR,"Fatal: grab device does not receive any video signal\n");

            }

            goto fail;

        }

        for (j = 1; j < s->gb_buffers.frames; j++) {

          s->gb_buf.frame = j;

          ioctl(video_fd, VIDIOCMCAPTURE, &s->gb_buf);

        }

        s->frame_format = s->gb_buf.format;

        s->use_mmap = 1;

    }



    for (j = 0; j < vformat_num; j++) {

        if (s->frame_format == video_formats[j].palette) {

            frame_size = width * height * video_formats[j].depth / 8;

            st->codec->pix_fmt = video_formats[j].pix_fmt;

            break;

        }

    }



    if (j >= vformat_num)

        goto fail;



    s->fd = video_fd;

    s->frame_size = frame_size;



    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_RAWVIDEO;

    st->codec->width = width;

    st->codec->height = height;

    st->codec->time_base.den      = frame_rate;

    st->codec->time_base.num = frame_rate_base;

    st->codec->bit_rate = frame_size * 1/av_q2d(st->codec->time_base) * 8;



    return 0;

 fail:

    if (video_fd >= 0)

        close(video_fd);

    av_free(st);

    return AVERROR(EIO);

}
