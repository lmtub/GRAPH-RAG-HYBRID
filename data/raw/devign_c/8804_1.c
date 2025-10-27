AVStream *avformat_new_stream(AVFormatContext *s, const AVCodec *c)

{

    AVStream *st;

    int i;

    AVStream **streams;



    if (s->nb_streams >= INT_MAX/sizeof(*streams))


    streams = av_realloc_array(s->streams, s->nb_streams + 1, sizeof(*streams));

    if (!streams)


    s->streams = streams;



    st = av_mallocz(sizeof(AVStream));

    if (!st)


    if (!(st->info = av_mallocz(sizeof(*st->info)))) {




    st->info->last_dts = AV_NOPTS_VALUE;



    st->codec = avcodec_alloc_context3(c);






    if (s->iformat) {

        /* no default bitrate if decoding */

        st->codec->bit_rate = 0;



        /* default pts setting is MPEG-like */

        avpriv_set_pts_info(st, 33, 1, 90000);




    st->index      = s->nb_streams;

    st->start_time = AV_NOPTS_VALUE;

    st->duration   = AV_NOPTS_VALUE;

    /* we set the current DTS to 0 so that formats without any timestamps

     * but durations get some timestamps, formats with some unknown

     * timestamps have their first few packets buffered and the

     * timestamps corrected before they are returned to the user */

    st->cur_dts       = s->iformat ? RELATIVE_TS_BASE : 0;

    st->first_dts     = AV_NOPTS_VALUE;

    st->probe_packets = MAX_PROBE_PACKETS;

    st->pts_wrap_reference = AV_NOPTS_VALUE;

    st->pts_wrap_behavior = AV_PTS_WRAP_IGNORE;



    st->last_IP_pts = AV_NOPTS_VALUE;

    st->last_dts_for_order_check = AV_NOPTS_VALUE;

    for (i = 0; i < MAX_REORDER_DELAY + 1; i++)

        st->pts_buffer[i] = AV_NOPTS_VALUE;



    st->sample_aspect_ratio = (AVRational) { 0, 1 };



#if FF_API_R_FRAME_RATE

    st->info->last_dts      = AV_NOPTS_VALUE;

#endif

    st->info->fps_first_dts = AV_NOPTS_VALUE;

    st->info->fps_last_dts  = AV_NOPTS_VALUE;



    st->inject_global_side_data = s->internal->inject_global_side_data;



    s->streams[s->nb_streams++] = st;

    return st;
