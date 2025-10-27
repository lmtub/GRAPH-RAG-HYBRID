AVStream *av_new_stream(AVFormatContext *s, int id)

{

    AVStream *st;

    int i;

#if LIBAVFORMAT_VERSION_MAJOR >= 53

    AVStream **streams;



    if (s->nb_streams >= INT_MAX/sizeof(*streams))

        return NULL;

    streams = av_realloc(s->streams, (s->nb_streams + 1) * sizeof(*streams));

    if (!streams)

        return NULL;

    s->streams = streams;

#else

    if (s->nb_streams >= MAX_STREAMS){

        av_log(s, AV_LOG_ERROR, "Too many streams\n");

        return NULL;

    }

#endif



    st = av_mallocz(sizeof(AVStream));

    if (!st)

        return NULL;



    st->codec= avcodec_alloc_context();

    if (s->iformat) {

        /* no default bitrate if decoding */

        st->codec->bit_rate = 0;

    }

    st->index = s->nb_streams;

    st->id = id;

    st->start_time = AV_NOPTS_VALUE;

    st->duration = AV_NOPTS_VALUE;

        /* we set the current DTS to 0 so that formats without any timestamps

           but durations get some timestamps, formats with some unknown

           timestamps have their first few packets buffered and the

           timestamps corrected before they are returned to the user */

    st->cur_dts = 0;

    st->first_dts = AV_NOPTS_VALUE;

    st->probe_packets = MAX_PROBE_PACKETS;



    /* default pts setting is MPEG-like */

    av_set_pts_info(st, 33, 1, 90000);

    st->last_IP_pts = AV_NOPTS_VALUE;

    for(i=0; i<MAX_REORDER_DELAY+1; i++)

        st->pts_buffer[i]= AV_NOPTS_VALUE;

    st->reference_dts = AV_NOPTS_VALUE;



    st->sample_aspect_ratio = (AVRational){0,1};



    s->streams[s->nb_streams++] = st;

    return st;

}
