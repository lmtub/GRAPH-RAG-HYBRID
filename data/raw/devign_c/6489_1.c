static int video_thread(void *arg)

{

    VideoState *is = arg;

    AVFrame *frame = avcodec_alloc_frame();

    int64_t pts_int;

    double pts;

    int ret;



#if CONFIG_AVFILTER

    AVFilterGraph *graph = avfilter_graph_alloc();

    AVFilterContext *filt_out = NULL;

    int64_t pos;

    int last_w = is->video_st->codec->width;

    int last_h = is->video_st->codec->height;



    if ((ret = configure_video_filters(graph, is, vfilters)) < 0)

        goto the_end;

    filt_out = is->out_video_filter;

#endif



    for (;;) {

#if !CONFIG_AVFILTER

        AVPacket pkt;

#else

        AVFilterBufferRef *picref;

        AVRational tb;

#endif

        while (is->paused && !is->videoq.abort_request)

            SDL_Delay(10);

#if CONFIG_AVFILTER

        if (   last_w != is->video_st->codec->width

            || last_h != is->video_st->codec->height) {

            av_dlog(NULL, "Changing size %dx%d -> %dx%d\n", last_w, last_h,

                    is->video_st->codec->width, is->video_st->codec->height);

            avfilter_graph_free(&graph);

            graph = avfilter_graph_alloc();

            if ((ret = configure_video_filters(graph, is, vfilters)) < 0)

                goto the_end;

            filt_out = is->out_video_filter;

            last_w = is->video_st->codec->width;

            last_h = is->video_st->codec->height;

        }

        ret = get_filtered_video_frame(filt_out, frame, &picref, &tb);

        if (picref) {

            pts_int = picref->pts;

            pos     = picref->pos;

            frame->opaque = picref;

        }



        if (av_cmp_q(tb, is->video_st->time_base)) {

            av_unused int64_t pts1 = pts_int;

            pts_int = av_rescale_q(pts_int, tb, is->video_st->time_base);

            av_dlog(NULL, "video_thread(): "

                    "tb:%d/%d pts:%"PRId64" -> tb:%d/%d pts:%"PRId64"\n",

                    tb.num, tb.den, pts1,

                    is->video_st->time_base.num, is->video_st->time_base.den, pts_int);

        }

#else

        ret = get_video_frame(is, frame, &pts_int, &pkt);

#endif



        if (ret < 0)

            goto the_end;



        if (!ret)

            continue;



        pts = pts_int * av_q2d(is->video_st->time_base);



#if CONFIG_AVFILTER

        ret = output_picture2(is, frame, pts, pos);

#else

        ret = output_picture2(is, frame, pts,  pkt.pos);

        av_free_packet(&pkt);

#endif

        if (ret < 0)

            goto the_end;



        if (step)

            if (cur_stream)

                stream_pause(cur_stream);

    }

 the_end:

#if CONFIG_AVFILTER

    avfilter_graph_free(&graph);

#endif

    av_free(frame);

    return 0;

}
