static int device_init(AVFormatContext *ctx, int *width, int *height,

                       uint32_t pix_fmt)

{

    struct video_data *s = ctx->priv_data;

    int fd = s->fd;

    struct v4l2_format fmt;

    struct v4l2_pix_format *pix = &fmt.fmt.pix;



    int res;



    memset(&fmt, 0, sizeof(struct v4l2_format));



    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    pix->width = *width;

    pix->height = *height;

    pix->pixelformat = pix_fmt;

    pix->field = V4L2_FIELD_ANY;



    res = ioctl(fd, VIDIOC_S_FMT, &fmt);



    if ((*width != fmt.fmt.pix.width) || (*height != fmt.fmt.pix.height)) {

        av_log(ctx, AV_LOG_INFO,

               "The V4L2 driver changed the video from %dx%d to %dx%d\n",

               *width, *height, fmt.fmt.pix.width, fmt.fmt.pix.height);

        *width = fmt.fmt.pix.width;

        *height = fmt.fmt.pix.height;

    }



    if (pix_fmt != fmt.fmt.pix.pixelformat) {

        av_log(ctx, AV_LOG_DEBUG,

               "The V4L2 driver changed the pixel format "

               "from 0x%08X to 0x%08X\n",

               pix_fmt, fmt.fmt.pix.pixelformat);

        res = -1;

    }



    if (fmt.fmt.pix.field == V4L2_FIELD_INTERLACED) {

        av_log(ctx, AV_LOG_DEBUG, "The V4L2 driver using the interlaced mode");

        s->interlaced = 1;

    }



    return res;

}
