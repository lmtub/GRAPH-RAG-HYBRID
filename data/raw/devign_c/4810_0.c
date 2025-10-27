static int x11grab_read_packet(AVFormatContext *s1, AVPacket *pkt)

{

    X11GrabContext *s = s1->priv_data;

    Display *dpy      = s->dpy;

    XImage *image     = s->image;

    int x_off         = s->x_off;

    int y_off         = s->y_off;

    int follow_mouse  = s->follow_mouse;

    int screen;

    Window root;

    int64_t curtime, delay;

    struct timespec ts;



    /* Calculate the time of the next frame */

    s->time_frame += INT64_C(1000000);



    /* wait based on the frame rate */

    for (;;) {

        curtime = av_gettime();

        delay   = s->time_frame * av_q2d(s->time_base) - curtime;

        if (delay <= 0) {

            if (delay < INT64_C(-1000000) * av_q2d(s->time_base))

                s->time_frame += INT64_C(1000000);

            break;

        }

        ts.tv_sec  = delay / 1000000;

        ts.tv_nsec = (delay % 1000000) * 1000;

        nanosleep(&ts, NULL);

    }



    av_init_packet(pkt);

    pkt->data = image->data;

    pkt->size = s->frame_size;

    pkt->pts  = curtime;



    screen = DefaultScreen(dpy);

    root   = RootWindow(dpy, screen);

    if (follow_mouse) {

        int screen_w, screen_h;

        int pointer_x, pointer_y, _;

        Window w;



        screen_w = DisplayWidth(dpy, screen);

        screen_h = DisplayHeight(dpy, screen);

        XQueryPointer(dpy, root, &w, &w, &pointer_x, &pointer_y, &_, &_, &_);

        if (follow_mouse == -1) {

            // follow the mouse, put it at center of grabbing region

            x_off += pointer_x - s->width / 2 - x_off;

            y_off += pointer_y - s->height / 2 - y_off;

        } else {

            // follow the mouse, but only move the grabbing region when mouse

            // reaches within certain pixels to the edge.

            if (pointer_x > x_off + s->width - follow_mouse)

                x_off += pointer_x - (x_off + s->width - follow_mouse);

            else if (pointer_x < x_off + follow_mouse)

                x_off -= (x_off + follow_mouse) - pointer_x;

            if (pointer_y > y_off + s->height - follow_mouse)

                y_off += pointer_y - (y_off + s->height - follow_mouse);

            else if (pointer_y < y_off + follow_mouse)

                y_off -= (y_off + follow_mouse) - pointer_y;

        }

        // adjust grabbing region position if it goes out of screen.

        s->x_off = x_off = FFMIN(FFMAX(x_off, 0), screen_w - s->width);

        s->y_off = y_off = FFMIN(FFMAX(y_off, 0), screen_h - s->height);



        if (s->show_region && s->region_win)

            XMoveWindow(dpy, s->region_win,

                        s->x_off - REGION_WIN_BORDER,

                        s->y_off - REGION_WIN_BORDER);

    }



    if (s->show_region) {

        if (s->region_win) {

            XEvent evt = { .type = NoEventMask };

            // Clean up the events, and do the initial draw or redraw.

            while (XCheckMaskEvent(dpy, ExposureMask | StructureNotifyMask,

                                   &evt))

                ;

            if (evt.type)

                x11grab_draw_region_win(s);

        } else {

            x11grab_region_win_init(s);

        }

    }



    if (s->use_shm) {

        if (!XShmGetImage(dpy, root, image, x_off, y_off, AllPlanes))

            av_log(s1, AV_LOG_INFO, "XShmGetImage() failed\n");

    } else {

        if (!xget_zpixmap(dpy, root, image, x_off, y_off))

            av_log(s1, AV_LOG_INFO, "XGetZPixmap() failed\n");

    }



    if (s->draw_mouse)

        paint_mouse_pointer(image, s);



    return s->frame_size;

}
