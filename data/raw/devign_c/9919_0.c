static int mmap_start(AVFormatContext *ctx)

{

    struct video_data *s = ctx->priv_data;

    enum v4l2_buf_type type;

    int i, res;



    for (i = 0; i < s->buffers; i++) {

        struct v4l2_buffer buf;



        memset(&buf, 0, sizeof(struct v4l2_buffer));

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        buf.memory = V4L2_MEMORY_MMAP;

        buf.index  = i;



        res = ioctl(s->fd, VIDIOC_QBUF, &buf);

        if (res < 0) {

            av_log(ctx, AV_LOG_ERROR, "ioctl(VIDIOC_QBUF): %s\n",

                   strerror(errno));



            return AVERROR(errno);

        }

    }



    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    res = ioctl(s->fd, VIDIOC_STREAMON, &type);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "ioctl(VIDIOC_STREAMON): %s\n",

               strerror(errno));



        return AVERROR(errno);

    }



    return 0;

}
