x11grab_read_packet(AVFormatContext *s1, AVPacket *pkt)

{

    struct x11_grab *s = s1->priv_data;

    Display *dpy = s->dpy;

    XImage *image = s->image;

    int x_off = s->x_off;

    int y_off = s->y_off;



    int64_t curtime, delay;

    struct timespec ts;



    /* Calculate the time of the next frame */

    s->time_frame += INT64_C(1000000);



    /* wait based on the frame rate */

    for(;;) {

        curtime = av_gettime();

        delay = s->time_frame * av_q2d(s->time_base) - curtime;

        if (delay <= 0) {

            if (delay < INT64_C(-1000000) * av_q2d(s->time_base)) {

                s->time_frame += INT64_C(1000000);

            }

            break;

        }

        ts.tv_sec = delay / 1000000;

        ts.tv_nsec = (delay % 1000000) * 1000;

        nanosleep(&ts, NULL);

    }



    av_init_packet(pkt);

    pkt->data = image->data;

    pkt->size = s->frame_size;

    pkt->pts = curtime;



    if(s->use_shm) {

        if (!XShmGetImage(dpy, RootWindow(dpy, DefaultScreen(dpy)), image, x_off, y_off, AllPlanes)) {

            av_log (s1, AV_LOG_INFO, "XShmGetImage() failed\n");

        }

    } else {

        if (!xget_zpixmap(dpy, RootWindow(dpy, DefaultScreen(dpy)), image, x_off, y_off)) {

            av_log (s1, AV_LOG_INFO, "XGetZPixmap() failed\n");

        }

    }



    if(!s->nomouse){

        paint_mouse_pointer(image, s);

    }



    return s->frame_size;

}
