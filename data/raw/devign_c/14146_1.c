void ff_mpegts_parse_close(MpegTSContext *ts)

{

    int i;



    for(i=0;i<NB_PID_MAX;i++)

        av_free(ts->pids[i]);

    av_free(ts);

}
