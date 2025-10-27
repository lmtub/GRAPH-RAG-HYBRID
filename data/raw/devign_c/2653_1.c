int ff_ass_add_rect(AVSubtitle *sub, const char *dialog,

                    int ts_start, int duration, int raw)

{

    AVBPrint buf;

    int ret, dlen;

    AVSubtitleRect **rects;



    av_bprint_init(&buf, 0, AV_BPRINT_SIZE_UNLIMITED);

    if ((ret = ff_ass_bprint_dialog(&buf, dialog, ts_start, duration, raw)) < 0)

        goto err;

    dlen = ret;

    if (!av_bprint_is_complete(&buf))




    rects = av_realloc_array(sub->rects, (sub->num_rects+1), sizeof(*sub->rects));

    if (!rects)


    sub->rects = rects;

    sub->end_display_time = FFMAX(sub->end_display_time, 10 * duration);

    rects[sub->num_rects]       = av_mallocz(sizeof(*rects[0]));



    rects[sub->num_rects]->type = SUBTITLE_ASS;

    ret = av_bprint_finalize(&buf, &rects[sub->num_rects]->ass);

    if (ret < 0)

        goto err;

    sub->num_rects++;

    return dlen;



errnomem:

    ret = AVERROR(ENOMEM);

err:

    av_bprint_finalize(&buf, NULL);

    return ret;

}