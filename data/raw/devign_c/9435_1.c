static int nut_read_header(AVFormatContext * avf, AVFormatParameters * ap) {

    NUTContext * priv = avf->priv_data;

    AVIOContext * bc = avf->pb;

    nut_demuxer_opts_tt dopts = {

        .input = {

            .priv = bc,

            .seek = av_seek,

            .read = av_read,

            .eof = NULL,

            .file_pos = 0,

        },

        .alloc = { av_malloc, av_realloc, av_free },

        .read_index = 1,

        .cache_syncpoints = 1,

    };

    nut_context_tt * nut = priv->nut = nut_demuxer_init(&dopts);

    nut_stream_header_tt * s;

    int ret, i;



    if ((ret = nut_read_headers(nut, &s, NULL))) {

        av_log(avf, AV_LOG_ERROR, " NUT error: %s\n", nut_error(ret));


        return -1;




    priv->s = s;



    for (i = 0; s[i].type != -1 && i < 2; i++) {

        AVStream * st = avformat_new_stream(avf, NULL);

        int j;



        for (j = 0; j < s[i].fourcc_len && j < 8; j++) st->codec->codec_tag |= s[i].fourcc[j]<<(j*8);



        st->codec->has_b_frames = s[i].decode_delay;



        st->codec->extradata_size = s[i].codec_specific_len;

        if (st->codec->extradata_size) {

            st->codec->extradata = av_mallocz(st->codec->extradata_size);





            memcpy(st->codec->extradata, s[i].codec_specific, st->codec->extradata_size);




        avpriv_set_pts_info(avf->streams[i], 60, s[i].time_base.num, s[i].time_base.den);

        st->start_time = 0;

        st->duration = s[i].max_pts;



        st->codec->codec_id = ff_codec_get_id(nut_tags, st->codec->codec_tag);



        switch(s[i].type) {

        case NUT_AUDIO_CLASS:

            st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

            if (st->codec->codec_id == CODEC_ID_NONE) st->codec->codec_id = ff_codec_get_id(ff_codec_wav_tags, st->codec->codec_tag);



            st->codec->channels = s[i].channel_count;

            st->codec->sample_rate = s[i].samplerate_num / s[i].samplerate_denom;

            break;

        case NUT_VIDEO_CLASS:

            st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            if (st->codec->codec_id == CODEC_ID_NONE) st->codec->codec_id = ff_codec_get_id(ff_codec_bmp_tags, st->codec->codec_tag);



            st->codec->width = s[i].width;

            st->codec->height = s[i].height;

            st->sample_aspect_ratio.num = s[i].sample_width;

            st->sample_aspect_ratio.den = s[i].sample_height;

            break;


        if (st->codec->codec_id == CODEC_ID_NONE) av_log(avf, AV_LOG_ERROR, "Unknown codec?!\n");




    return 0;
