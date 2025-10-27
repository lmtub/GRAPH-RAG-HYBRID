static int xv_write_header(AVFormatContext *s)

{

    XVContext *xv = s->priv_data;

    unsigned int num_adaptors;

    XvAdaptorInfo *ai;

    XvImageFormatValues *fv;

    int num_formats = 0, j;

    AVCodecContext *encctx = s->streams[0]->codec;



    if (   s->nb_streams > 1

        || encctx->codec_type != AVMEDIA_TYPE_VIDEO

        || encctx->codec_id   != AV_CODEC_ID_RAWVIDEO) {

        av_log(s, AV_LOG_ERROR, "Only supports one rawvideo stream\n");

        return AVERROR(EINVAL);

    }



    xv->display = XOpenDisplay(xv->display_name);

    if (!xv->display) {

        av_log(s, AV_LOG_ERROR, "Could not open the X11 display '%s'\n", xv->display_name);

        return AVERROR(EINVAL);

    }



    xv->image_width  = encctx->width;

    xv->image_height = encctx->height;

    if (!xv->window_width && !xv->window_height) {

        xv->window_width  = encctx->width;

        xv->window_height = encctx->height;

    }

    xv->window = XCreateSimpleWindow(xv->display, DefaultRootWindow(xv->display),

                                     xv->window_x, xv->window_y,

                                     xv->window_width, xv->window_height,

                                     0, 0, 0);

    if (!xv->window_title) {

        if (!(xv->window_title = av_strdup(s->filename)))

            return AVERROR(ENOMEM);

    }

    XStoreName(xv->display, xv->window, xv->window_title);

    XMapWindow(xv->display, xv->window);



    if (XvQueryAdaptors(xv->display, DefaultRootWindow(xv->display), &num_adaptors, &ai) != Success)

        return AVERROR_EXTERNAL;

    xv->xv_port = ai[0].base_id;




    if (encctx->pix_fmt != AV_PIX_FMT_YUV420P) {

        av_log(s, AV_LOG_ERROR,

               "Unsupported pixel format '%s', only yuv420p is currently supported\n",

               av_get_pix_fmt_name(encctx->pix_fmt));

        return AVERROR_PATCHWELCOME;

    }



    fv = XvListImageFormats(xv->display, xv->xv_port, &num_formats);

    if (!fv)

        return AVERROR_EXTERNAL;

    for (j = 0; j < num_formats; j++) {

        if (fv[j].id == MKTAG('I','4','2','0')) {

            break;

        }

    }

    XFree(fv);



    if (j >= num_formats) {

        av_log(s, AV_LOG_ERROR,

               "Device does not support pixel format yuv420p, aborting\n");

        return AVERROR(EINVAL);

    }



    xv->gc = XCreateGC(xv->display, xv->window, 0, 0);

    xv->image_width  = encctx->width;

    xv->image_height = encctx->height;

    xv->yuv_image = XvShmCreateImage(xv->display, xv->xv_port,

                                     MKTAG('I','4','2','0'), 0,

                                     xv->image_width, xv->image_height, &xv->yuv_shminfo);

    xv->yuv_shminfo.shmid = shmget(IPC_PRIVATE, xv->yuv_image->data_size,

                                   IPC_CREAT | 0777);

    xv->yuv_shminfo.shmaddr = (char *)shmat(xv->yuv_shminfo.shmid, 0, 0);

    xv->yuv_image->data = xv->yuv_shminfo.shmaddr;

    xv->yuv_shminfo.readOnly = False;



    XShmAttach(xv->display, &xv->yuv_shminfo);

    XSync(xv->display, False);

    shmctl(xv->yuv_shminfo.shmid, IPC_RMID, 0);



    return 0;

}