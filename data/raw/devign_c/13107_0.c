static void video_image_display(VideoState *is)

{

    Frame *vp;

    Frame *sp;

    AVPicture pict;

    SDL_Rect rect;

    int i;



    vp = frame_queue_peek(&is->pictq);

    if (vp->bmp) {

        if (is->subtitle_st) {

            if (frame_queue_nb_remaining(&is->subpq) > 0) {

                sp = frame_queue_peek(&is->subpq);



                if (vp->pts >= sp->pts + ((float) sp->sub.start_display_time / 1000)) {

                    SDL_LockYUVOverlay (vp->bmp);



                    pict.data[0] = vp->bmp->pixels[0];

                    pict.data[1] = vp->bmp->pixels[2];

                    pict.data[2] = vp->bmp->pixels[1];



                    pict.linesize[0] = vp->bmp->pitches[0];

                    pict.linesize[1] = vp->bmp->pitches[2];

                    pict.linesize[2] = vp->bmp->pitches[1];



                    for (i = 0; i < sp->sub.num_rects; i++)

                        blend_subrect(&pict, sp->sub.rects[i],

                                      vp->bmp->w, vp->bmp->h);



                    SDL_UnlockYUVOverlay (vp->bmp);

                }

            }

        }



        calculate_display_rect(&rect, is->xleft, is->ytop, is->width, is->height, vp->width, vp->height, vp->sar);



        SDL_DisplayYUVOverlay(vp->bmp, &rect);



        if (rect.x != is->last_display_rect.x || rect.y != is->last_display_rect.y || rect.w != is->last_display_rect.w || rect.h != is->last_display_rect.h || is->force_refresh) {

            int bgcolor = SDL_MapRGB(screen->format, 0x00, 0x00, 0x00);

            fill_border(is->xleft, is->ytop, is->width, is->height, rect.x, rect.y, rect.w, rect.h, bgcolor, 1);

            is->last_display_rect = rect;

        }

    }

}
