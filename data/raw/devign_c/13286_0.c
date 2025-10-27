static int subtitle_thread(void *arg)

{

    VideoState *is = arg;

    Frame *sp;

    int got_subtitle;

    double pts;

    int i;



    for (;;) {

        if (!(sp = frame_queue_peek_writable(&is->subpq)))

            return 0;



        if ((got_subtitle = decoder_decode_frame(&is->subdec, NULL, &sp->sub)) < 0)

            break;



        pts = 0;



        if (got_subtitle && sp->sub.format == 0) {

            if (sp->sub.pts != AV_NOPTS_VALUE)

                pts = sp->sub.pts / (double)AV_TIME_BASE;

            sp->pts = pts;

            sp->serial = is->subdec.pkt_serial;



            for (i = 0; i < sp->sub.num_rects; i++)

            {

                int in_w = sp->sub.rects[i]->w;

                int in_h = sp->sub.rects[i]->h;

                int subw = is->subdec.avctx->width  ? is->subdec.avctx->width  : is->viddec_width;

                int subh = is->subdec.avctx->height ? is->subdec.avctx->height : is->viddec_height;

                int out_w = is->viddec_width  ? in_w * is->viddec_width  / subw : in_w;

                int out_h = is->viddec_height ? in_h * is->viddec_height / subh : in_h;

                AVPicture newpic;



                //can not use avpicture_alloc as it is not compatible with avsubtitle_free()

                av_image_fill_linesizes(newpic.linesize, AV_PIX_FMT_YUVA420P, out_w);

                newpic.data[0] = av_malloc(newpic.linesize[0] * out_h);

                newpic.data[3] = av_malloc(newpic.linesize[3] * out_h);

                newpic.data[1] = av_malloc(newpic.linesize[1] * ((out_h+1)/2));

                newpic.data[2] = av_malloc(newpic.linesize[2] * ((out_h+1)/2));



                is->sub_convert_ctx = sws_getCachedContext(is->sub_convert_ctx,

                    in_w, in_h, AV_PIX_FMT_PAL8, out_w, out_h,

                    AV_PIX_FMT_YUVA420P, sws_flags, NULL, NULL, NULL);

                if (!is->sub_convert_ctx || !newpic.data[0] || !newpic.data[3] ||

                    !newpic.data[1] || !newpic.data[2]

                ) {

                    av_log(NULL, AV_LOG_FATAL, "Cannot initialize the sub conversion context\n");

                    exit(1);

                }

                sws_scale(is->sub_convert_ctx,

                          (void*)sp->sub.rects[i]->pict.data, sp->sub.rects[i]->pict.linesize,

                          0, in_h, newpic.data, newpic.linesize);



                av_free(sp->sub.rects[i]->pict.data[0]);

                av_free(sp->sub.rects[i]->pict.data[1]);

                sp->sub.rects[i]->pict = newpic;

                sp->sub.rects[i]->w = out_w;

                sp->sub.rects[i]->h = out_h;

                sp->sub.rects[i]->x = sp->sub.rects[i]->x * out_w / in_w;

                sp->sub.rects[i]->y = sp->sub.rects[i]->y * out_h / in_h;

            }



            /* now we can update the picture count */

            frame_queue_push(&is->subpq);

        } else if (got_subtitle) {

            avsubtitle_free(&sp->sub);

        }

    }

    return 0;

}
