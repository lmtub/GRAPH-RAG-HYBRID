static void stream_close(VideoState *is)

{

    VideoPicture *vp;

    int i;

    /* XXX: use a special url_shutdown call to abort parse cleanly */

    is->abort_request = 1;

    SDL_WaitThread(is->read_tid, NULL);

    SDL_WaitThread(is->refresh_tid, NULL);

    packet_queue_destroy(&is->videoq);

    packet_queue_destroy(&is->audioq);

    packet_queue_destroy(&is->subtitleq);



    /* free all pictures */

    for (i = 0; i < VIDEO_PICTURE_QUEUE_SIZE; i++) {

        vp = &is->pictq[i];

#if CONFIG_AVFILTER

        avfilter_unref_bufferp(&vp->picref);

#endif

        if (vp->bmp) {

            SDL_FreeYUVOverlay(vp->bmp);

            vp->bmp = NULL;

        }

    }

    SDL_DestroyMutex(is->pictq_mutex);

    SDL_DestroyCond(is->pictq_cond);

    SDL_DestroyMutex(is->subpq_mutex);

    SDL_DestroyCond(is->subpq_cond);

    SDL_DestroyCond(is->continue_read_thread);

#if !CONFIG_AVFILTER

    sws_freeContext(is->img_convert_ctx);

#endif

    av_free(is);

}
