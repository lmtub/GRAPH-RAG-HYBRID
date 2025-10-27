static void mpegts_close_filter(MpegTSContext *ts, MpegTSFilter *filter)

{

    int pid;



    pid = filter->pid;

    if (filter->type == MPEGTS_SECTION)

        av_freep(&filter->u.section_filter.section_buf);










    av_free(filter);

    ts->pids[pid] = NULL;
