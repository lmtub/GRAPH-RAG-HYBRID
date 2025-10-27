static int mmap_init(AVFormatContext *ctx)

{

    int i, res;

    struct video_data *s = ctx->priv_data;

    struct v4l2_requestbuffers req = {

        .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,

        .count  = desired_video_buffers,

        .memory = V4L2_MEMORY_MMAP

    };



    res = ioctl(s->fd, VIDIOC_REQBUFS, &req);

    if (res < 0) {

        if (errno == EINVAL) {

            av_log(ctx, AV_LOG_ERROR, "Device does not support mmap\n");

        } else {

            av_log(ctx, AV_LOG_ERROR, "ioctl(VIDIOC_REQBUFS)\n");

        }



        return AVERROR(errno);

    }



    if (req.count < 2) {

        av_log(ctx, AV_LOG_ERROR, "Insufficient buffer memory\n");



        return AVERROR(ENOMEM);

    }

    s->buffers = req.count;

    s->buf_start = av_malloc(sizeof(void *) * s->buffers);

    if (s->buf_start == NULL) {

        av_log(ctx, AV_LOG_ERROR, "Cannot allocate buffer pointers\n");



        return AVERROR(ENOMEM);

    }

    s->buf_len = av_malloc(sizeof(unsigned int) * s->buffers);

    if (s->buf_len == NULL) {

        av_log(ctx, AV_LOG_ERROR, "Cannot allocate buffer sizes\n");

        av_free(s->buf_start);



        return AVERROR(ENOMEM);

    }



    for (i = 0; i < req.count; i++) {

        struct v4l2_buffer buf = {

            .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,

            .index  = i,

            .memory = V4L2_MEMORY_MMAP

        };



        res = ioctl(s->fd, VIDIOC_QUERYBUF, &buf);

        if (res < 0) {

            av_log(ctx, AV_LOG_ERROR, "ioctl(VIDIOC_QUERYBUF)\n");



            return AVERROR(errno);

        }



        s->buf_len[i] = buf.length;

        if (s->frame_size > 0 && s->buf_len[i] < s->frame_size) {

            av_log(ctx, AV_LOG_ERROR,

                   "Buffer len [%d] = %d != %d\n",

                   i, s->buf_len[i], s->frame_size);



            return -1;

        }

        s->buf_start[i] = mmap(NULL, buf.length,

                               PROT_READ | PROT_WRITE, MAP_SHARED,

                               s->fd, buf.m.offset);



        if (s->buf_start[i] == MAP_FAILED) {

            av_log(ctx, AV_LOG_ERROR, "mmap: %s\n", strerror(errno));



            return AVERROR(errno);

        }

    }



    return 0;

}
