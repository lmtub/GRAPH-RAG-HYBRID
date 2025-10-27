static int convert_sub_to_old_ass_form(AVSubtitle *sub, const AVPacket *pkt, AVRational tb)

{

    int i;

    AVBPrint buf;



    av_bprint_init(&buf, 0, AV_BPRINT_SIZE_UNLIMITED);



    for (i = 0; i < sub->num_rects; i++) {

        char *final_dialog;

        const char *dialog;

        AVSubtitleRect *rect = sub->rects[i];

        int ts_start, ts_duration = -1;

        long int layer;



        if (rect->type != SUBTITLE_ASS || !strncmp(rect->ass, "Dialogue ", 10))

            continue;



        av_bprint_clear(&buf);



        /* skip ReadOrder */

        dialog = strchr(rect->ass, ',');

        if (!dialog)

            continue;

        dialog++;



        /* extract Layer or Marked */

        layer = strtol(dialog, (char**)&dialog, 10);

        if (*dialog != ',')

            continue;

        dialog++;



        /* rescale timing to ASS time base (ms) */

        ts_start = av_rescale_q(pkt->pts, tb, av_make_q(1, 100));

        if (pkt->duration != -1)

            ts_duration = av_rescale_q(pkt->duration, tb, av_make_q(1, 100));

        sub->end_display_time = FFMAX(sub->end_display_time, 10 * ts_duration);



        /* construct ASS (standalone file form with timestamps) string */

        av_bprintf(&buf, "Dialogue: %ld,", layer);

        insert_ts(&buf, ts_start);

        insert_ts(&buf, ts_duration == -1 ? -1 : ts_start + ts_duration);

        av_bprintf(&buf, "%s\r\n", dialog);



        final_dialog = av_strdup(buf.str);

        if (!av_bprint_is_complete(&buf) || !final_dialog) {

            av_freep(&final_dialog);

            av_bprint_finalize(&buf, NULL);

            return AVERROR(ENOMEM);

        }

        av_freep(&rect->ass);

        rect->ass = final_dialog;

    }



    av_bprint_finalize(&buf, NULL);

    return 0;

}
