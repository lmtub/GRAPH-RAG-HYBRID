static int device_open(AVFormatContext *ctx)

{

    struct v4l2_capability cap;

    int fd;

#if CONFIG_LIBV4L2

    int fd_libv4l;

#endif

    int res, err;

    int flags = O_RDWR;



    if (ctx->flags & AVFMT_FLAG_NONBLOCK) {

        flags |= O_NONBLOCK;

    }



    fd = v4l2_open(ctx->filename, flags, 0);

    if (fd < 0) {

        err = errno;



        av_log(ctx, AV_LOG_ERROR, "Cannot open video device %s : %s\n",

               ctx->filename, strerror(err));



        return AVERROR(err);

    }

#if CONFIG_LIBV4L2

    fd_libv4l = v4l2_fd_open(fd, 0);

    if (fd < 0) {

        err = AVERROR(errno);

        av_log(ctx, AV_LOG_ERROR, "Cannot open video device with libv4l neither %s : %s\n",

               ctx->filename, strerror(errno));

        return err;

    }

    fd = fd_libv4l;

#endif



    res = v4l2_ioctl(fd, VIDIOC_QUERYCAP, &cap);

    if (res < 0) {

        err = errno;

        av_log(ctx, AV_LOG_ERROR, "ioctl(VIDIOC_QUERYCAP): %s\n",

               strerror(err));



        goto fail;

    }



    av_log(ctx, AV_LOG_VERBOSE, "[%d]Capabilities: %x\n",

           fd, cap.capabilities);



    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {

        av_log(ctx, AV_LOG_ERROR, "Not a video capture device.\n");

        err = ENODEV;



        goto fail;

    }



    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {

        av_log(ctx, AV_LOG_ERROR,

               "The device does not support the streaming I/O method.\n");

        err = ENOSYS;



        goto fail;

    }



    return fd;



fail:

    v4l2_close(fd);

    return AVERROR(err);

}
