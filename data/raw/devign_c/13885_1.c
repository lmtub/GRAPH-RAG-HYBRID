static int v4l2_try_start(AVCodecContext *avctx)

{

    V4L2m2mContext *s = avctx->priv_data;

    V4L2Context *const capture = &s->capture;

    V4L2Context *const output = &s->output;

    struct v4l2_selection selection;

    int ret;



    /* 1. start the output process */

    if (!output->streamon) {

        ret = ff_v4l2_context_set_status(output, VIDIOC_STREAMON);

        if (ret < 0) {

            av_log(avctx, AV_LOG_DEBUG, "VIDIOC_STREAMON on output context\n");

            return ret;

        }

    }



    if (capture->streamon)

        return 0;



    /* 2. get the capture format */

    capture->format.type = capture->type;

    ret = ioctl(s->fd, VIDIOC_G_FMT, &capture->format);

    if (ret) {

        av_log(avctx, AV_LOG_WARNING, "VIDIOC_G_FMT ioctl\n");

        return ret;

    }



    /* 2.1 update the AVCodecContext */

    avctx->pix_fmt = ff_v4l2_format_v4l2_to_avfmt(capture->format.fmt.pix_mp.pixelformat, AV_CODEC_ID_RAWVIDEO);

    capture->av_pix_fmt = avctx->pix_fmt;



    /* 3. set the crop parameters */

    selection.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    selection.r.height = avctx->coded_height;

    selection.r.width = avctx->coded_width;

    ret = ioctl(s->fd, VIDIOC_S_SELECTION, &selection);

    if (!ret) {

        ret = ioctl(s->fd, VIDIOC_G_SELECTION, &selection);

        if (ret) {

            av_log(avctx, AV_LOG_WARNING, "VIDIOC_G_SELECTION ioctl\n");

        } else {

            av_log(avctx, AV_LOG_DEBUG, "crop output %dx%d\n", selection.r.width, selection.r.height);

            /* update the size of the resulting frame */

            capture->height = selection.r.height;

            capture->width  = selection.r.width;

        }

    }



    /* 4. init the capture context now that we have the capture format */

    if (!capture->buffers) {

        ret = ff_v4l2_context_init(capture);

        if (ret) {

            av_log(avctx, AV_LOG_DEBUG, "can't request output buffers\n");

            return ret;

        }

    }



    /* 5. start the capture process */

    ret = ff_v4l2_context_set_status(capture, VIDIOC_STREAMON);

    if (ret) {

        av_log(avctx, AV_LOG_DEBUG, "VIDIOC_STREAMON, on capture context\n");

        return ret;

    }



    return 0;

}
