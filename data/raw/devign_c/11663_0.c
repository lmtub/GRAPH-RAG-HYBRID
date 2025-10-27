static int queue_picture(VideoState *is, AVFrame *src_frame, double pts)

{

    VideoPicture *vp;

    int dst_pix_fmt;

    AVPicture pict;

    static struct SwsContext *img_convert_ctx;



    /* wait until we have space to put a new picture */

    SDL_LockMutex(is->pictq_mutex);

    while (is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE &&

           !is->videoq.abort_request) {

        SDL_CondWait(is->pictq_cond, is->pictq_mutex);

    }

    SDL_UnlockMutex(is->pictq_mutex);



    if (is->videoq.abort_request)

        return -1;



    vp = &is->pictq[is->pictq_windex];



    /* alloc or resize hardware picture buffer */

    if (!vp->bmp ||

        vp->width != is->video_st->codec->width ||

        vp->height != is->video_st->codec->height) {

        SDL_Event event;



        vp->allocated = 0;



        /* the allocation must be done in the main thread to avoid

           locking problems */

        event.type = FF_ALLOC_EVENT;

        event.user.data1 = is;

        SDL_PushEvent(&event);



        /* wait until the picture is allocated */

        SDL_LockMutex(is->pictq_mutex);

        while (!vp->allocated && !is->videoq.abort_request) {

            SDL_CondWait(is->pictq_cond, is->pictq_mutex);

        }

        SDL_UnlockMutex(is->pictq_mutex);



        if (is->videoq.abort_request)

            return -1;

    }



    /* if the frame is not skipped, then display it */

    if (vp->bmp) {

        /* get a pointer on the bitmap */

        SDL_LockYUVOverlay (vp->bmp);



        dst_pix_fmt = PIX_FMT_YUV420P;

        pict.data[0] = vp->bmp->pixels[0];

        pict.data[1] = vp->bmp->pixels[2];

        pict.data[2] = vp->bmp->pixels[1];



        pict.linesize[0] = vp->bmp->pitches[0];

        pict.linesize[1] = vp->bmp->pitches[2];

        pict.linesize[2] = vp->bmp->pitches[1];

        sws_flags = av_get_int(sws_opts, "sws_flags", NULL);

        img_convert_ctx = sws_getCachedContext(img_convert_ctx,

            is->video_st->codec->width, is->video_st->codec->height,

            is->video_st->codec->pix_fmt,

            is->video_st->codec->width, is->video_st->codec->height,

            dst_pix_fmt, sws_flags, NULL, NULL, NULL);

        if (img_convert_ctx == NULL) {

            fprintf(stderr, "Cannot initialize the conversion context\n");

            exit(1);

        }

        sws_scale(img_convert_ctx, src_frame->data, src_frame->linesize,

                  0, is->video_st->codec->height, pict.data, pict.linesize);

        /* update the bitmap content */

        SDL_UnlockYUVOverlay(vp->bmp);



        vp->pts = pts;



        /* now we can update the picture count */

        if (++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE)

            is->pictq_windex = 0;

        SDL_LockMutex(is->pictq_mutex);

        is->pictq_size++;

        SDL_UnlockMutex(is->pictq_mutex);

    }

    return 0;

}
