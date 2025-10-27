static int mpegts_read_close(AVFormatContext *s)

{

    MpegTSContext *ts = s->priv_data;

    int i;



    clear_programs(ts);



    for(i=0;i<NB_PID_MAX;i++)

        if (ts->pids[i]) mpegts_close_filter(ts, ts->pids[i]);



    return 0;

}
