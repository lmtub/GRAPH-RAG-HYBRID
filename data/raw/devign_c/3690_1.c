static void alloc_picture(void *opaque)

{

    VideoState *is = opaque;

    VideoPicture *vp;



    vp = &is->pictq[is->pictq_windex];



    if (vp->bmp)

        SDL_FreeYUVOverlay(vp->bmp);



#if CONFIG_AVFILTER

    if (vp->picref)

        avfilter_unref_buffer(vp->picref);

    vp->picref = NULL;



    vp->width   = is->out_video_filter->inputs[0]->w;

    vp->height  = is->out_video_filter->inputs[0]->h;

    vp->pix_fmt = is->out_video_filter->inputs[0]->format;

#else

    vp->width   = is->video_st->codec->width;

    vp->height  = is->video_st->codec->height;

    vp->pix_fmt = is->video_st->codec->pix_fmt;

#endif



    vp->bmp = SDL_CreateYUVOverlay(vp->width, vp->height,

                                   SDL_YV12_OVERLAY,

                                   screen);











    SDL_LockMutex(is->pictq_mutex);

    vp->allocated = 1;

    SDL_CondSignal(is->pictq_cond);

    SDL_UnlockMutex(is->pictq_mutex);
