static void alloc_picture(VideoState *is)

{

    VideoPicture *vp;

    int64_t bufferdiff;



    vp = &is->pictq[is->pictq_windex];



    if (vp->bmp)

        SDL_FreeYUVOverlay(vp->bmp);



    video_open(is, 0, vp);



    vp->bmp = SDL_CreateYUVOverlay(vp->width, vp->height,

                                   SDL_YV12_OVERLAY,

                                   screen);

    bufferdiff = vp->bmp ? FFMAX(vp->bmp->pixels[0], vp->bmp->pixels[1]) - FFMIN(vp->bmp->pixels[0], vp->bmp->pixels[1]) : 0;

    if (!vp->bmp || vp->bmp->pitches[0] < vp->width || bufferdiff < vp->height * vp->bmp->pitches[0]) {

        /* SDL allocates a buffer smaller than requested if the video

         * overlay hardware is unable to support the requested size. */

        av_log(NULL, AV_LOG_FATAL,

               "Error: the video system does not support an image\n"

                        "size of %dx%d pixels. Try using -lowres or -vf \"scale=w:h\"\n"

                        "to reduce the image size.\n", vp->width, vp->height );

        do_exit(is);

    }



    SDL_LockMutex(is->pictq_mutex);

    vp->allocated = 1;

    SDL_CondSignal(is->pictq_cond);

    SDL_UnlockMutex(is->pictq_mutex);

}
