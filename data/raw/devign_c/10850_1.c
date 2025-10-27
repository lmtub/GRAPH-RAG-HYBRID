x11grab_read_header(AVFormatContext *s1)

{

    struct x11_grab *x11grab = s1->priv_data;

    Display *dpy;

    AVStream *st = NULL;

    enum PixelFormat input_pixfmt;

    XImage *image;

    int x_off = 0;

    int y_off = 0;

    int screen;

    int use_shm;

    char *param, *offset;

    int ret = 0;

    AVRational framerate;



    param = av_strdup(s1->filename);

    if (!param)

        goto out;



    offset = strchr(param, '+');

    if (offset) {

        sscanf(offset, "%d,%d", &x_off, &y_off);

        x11grab->draw_mouse = !strstr(offset, "nomouse");

        *offset= 0;

    }



    if ((ret = av_parse_video_size(&x11grab->width, &x11grab->height, x11grab->video_size)) < 0) {

        av_log(s1, AV_LOG_ERROR, "Couldn't parse video size.\n");

        goto out;

    }

    if ((ret = av_parse_video_rate(&framerate, x11grab->framerate)) < 0) {

        av_log(s1, AV_LOG_ERROR, "Could not parse framerate: %s.\n", x11grab->framerate);

        goto out;

    }

    av_log(s1, AV_LOG_INFO, "device: %s -> display: %s x: %d y: %d width: %d height: %d\n",

           s1->filename, param, x_off, y_off, x11grab->width, x11grab->height);



    dpy = XOpenDisplay(param);

    if(!dpy) {

        av_log(s1, AV_LOG_ERROR, "Could not open X display.\n");

        ret = AVERROR(EIO);

        goto out;

    }



    st = avformat_new_stream(s1, NULL);

    if (!st) {

        ret = AVERROR(ENOMEM);

        goto out;

    }

    avpriv_set_pts_info(st, 64, 1, 1000000); /* 64 bits pts in us */



    screen = DefaultScreen(dpy);



    if (x11grab->follow_mouse) {

        int screen_w, screen_h;

        Window w;



        screen_w = DisplayWidth(dpy, screen);

        screen_h = DisplayHeight(dpy, screen);

        XQueryPointer(dpy, RootWindow(dpy, screen), &w, &w, &x_off, &y_off, &ret, &ret, &ret);

        x_off -= x11grab->width / 2;

        y_off -= x11grab->height / 2;

        x_off = FFMIN(FFMAX(x_off, 0), screen_w - x11grab->width);

        y_off = FFMIN(FFMAX(y_off, 0), screen_h - x11grab->height);

        av_log(s1, AV_LOG_INFO, "followmouse is enabled, resetting grabbing region to x: %d y: %d\n", x_off, y_off);

    }



    use_shm = XShmQueryExtension(dpy);

    av_log(s1, AV_LOG_INFO, "shared memory extension %s found\n", use_shm ? "" : "not");



    if(use_shm) {

        int scr = XDefaultScreen(dpy);

        image = XShmCreateImage(dpy,

                                DefaultVisual(dpy, scr),

                                DefaultDepth(dpy, scr),

                                ZPixmap,

                                NULL,

                                &x11grab->shminfo,

                                x11grab->width, x11grab->height);

        x11grab->shminfo.shmid = shmget(IPC_PRIVATE,

                                        image->bytes_per_line * image->height,

                                        IPC_CREAT|0777);

        if (x11grab->shminfo.shmid == -1) {

            av_log(s1, AV_LOG_ERROR, "Fatal: Can't get shared memory!\n");

            ret = AVERROR(ENOMEM);

            goto out;

        }

        x11grab->shminfo.shmaddr = image->data = shmat(x11grab->shminfo.shmid, 0, 0);

        x11grab->shminfo.readOnly = False;



        if (!XShmAttach(dpy, &x11grab->shminfo)) {

            av_log(s1, AV_LOG_ERROR, "Fatal: Failed to attach shared memory!\n");

            /* needs some better error subroutine :) */

            ret = AVERROR(EIO);

            goto out;

        }

    } else {

        image = XGetImage(dpy, RootWindow(dpy, screen),

                          x_off,y_off,

                          x11grab->width, x11grab->height,

                          AllPlanes, ZPixmap);

    }



    switch (image->bits_per_pixel) {

    case 8:

        av_log (s1, AV_LOG_DEBUG, "8 bit palette\n");

        input_pixfmt = PIX_FMT_PAL8;

        break;

    case 16:

        if (       image->red_mask   == 0xf800 &&

                   image->green_mask == 0x07e0 &&

                   image->blue_mask  == 0x001f ) {

            av_log (s1, AV_LOG_DEBUG, "16 bit RGB565\n");

            input_pixfmt = PIX_FMT_RGB565;

        } else if (image->red_mask   == 0x7c00 &&

                   image->green_mask == 0x03e0 &&

                   image->blue_mask  == 0x001f ) {

            av_log(s1, AV_LOG_DEBUG, "16 bit RGB555\n");

            input_pixfmt = PIX_FMT_RGB555;

        } else {

            av_log(s1, AV_LOG_ERROR, "RGB ordering at image depth %i not supported ... aborting\n", image->bits_per_pixel);

            av_log(s1, AV_LOG_ERROR, "color masks: r 0x%.6lx g 0x%.6lx b 0x%.6lx\n", image->red_mask, image->green_mask, image->blue_mask);

            ret = AVERROR(EIO);

            goto out;

        }

        break;

    case 24:

        if (        image->red_mask   == 0xff0000 &&

                    image->green_mask == 0x00ff00 &&

                    image->blue_mask  == 0x0000ff ) {

            input_pixfmt = PIX_FMT_BGR24;

        } else if ( image->red_mask   == 0x0000ff &&

                    image->green_mask == 0x00ff00 &&

                    image->blue_mask  == 0xff0000 ) {

            input_pixfmt = PIX_FMT_RGB24;

        } else {

            av_log(s1, AV_LOG_ERROR,"rgb ordering at image depth %i not supported ... aborting\n", image->bits_per_pixel);

            av_log(s1, AV_LOG_ERROR, "color masks: r 0x%.6lx g 0x%.6lx b 0x%.6lx\n", image->red_mask, image->green_mask, image->blue_mask);

            ret = AVERROR(EIO);

            goto out;

        }

        break;

    case 32:

        input_pixfmt = PIX_FMT_RGB32;

        break;

    default:

        av_log(s1, AV_LOG_ERROR, "image depth %i not supported ... aborting\n", image->bits_per_pixel);

        ret = AVERROR(EINVAL);

        goto out;

    }



    x11grab->frame_size = x11grab->width * x11grab->height * image->bits_per_pixel/8;

    x11grab->dpy = dpy;

    x11grab->time_base  = (AVRational){framerate.den, framerate.num};

    x11grab->time_frame = av_gettime() / av_q2d(x11grab->time_base);

    x11grab->x_off = x_off;

    x11grab->y_off = y_off;

    x11grab->image = image;

    x11grab->use_shm = use_shm;



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_RAWVIDEO;

    st->codec->width  = x11grab->width;

    st->codec->height = x11grab->height;

    st->codec->pix_fmt = input_pixfmt;

    st->codec->time_base = x11grab->time_base;

    st->codec->bit_rate = x11grab->frame_size * 1/av_q2d(x11grab->time_base) * 8;



out:


    return ret;

}