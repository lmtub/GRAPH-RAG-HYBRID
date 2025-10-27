static int nut_read_header(AVFormatContext * avf, AVFormatParameters * ap) {

    NUTContext * priv = avf->priv_data;

    ByteIOContext * bc = &avf->pb;

    nut_demuxer_opts_t dopts = {

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

    nut_context_t * nut = priv->nut = nut_demuxer_init(&dopts);

    nut_stream_header_t * s;

    int ret, i;



    if ((ret = nut_read_headers(nut, &s, NULL))) {

        if (ret < 0) av_log(avf, AV_LOG_ERROR, " NUT error: %s\n", nut_error(-ret));

        nut_demuxer_uninit(nut);

        return -1;

    }



    priv->s = s;



    for (i = 0; s[i].type != -1 && i < 2; i++) {

        AVStream * st = av_new_stream(avf, i);

        int j;



        for (j = 0; j < s[i].fourcc_len && j < 8; j++) st->codec->codec_tag |= s[i].fourcc[j]<<(j*8);



        st->codec->has_b_frames = s[i].decode_delay;



        st->codec->extradata_size = s[i].codec_specific_len;

        if (st->codec->extradata_size) {

            st->codec->extradata = av_mallocz(st->codec->extradata_size);

            memcpy(st->codec->extradata, s[i].codec_specific, st->codec->extradata_size);

        }



        av_set_pts_info(avf->streams[i], 60, s[i].time_base.nom, s[i].time_base.den);

        st->start_time = 0;

        st->duration = s[i].max_pts;



        st->codec->codec_id = codec_get_id(nut_tags, st->codec->codec_tag);



        switch(s[i].type) {

        case NUT_AUDIO_CLASS:

            st->codec->codec_type = CODEC_TYPE_AUDIO;

            if (st->codec->codec_id == CODEC_ID_NONE) st->codec->codec_id = codec_get_wav_id(st->codec->codec_tag);



            st->codec->channels = s[i].channel_count;

            st->codec->sample_rate = s[i].samplerate_nom / s[i].samplerate_denom;

            break;

        case NUT_VIDEO_CLASS:

            st->codec->codec_type = CODEC_TYPE_VIDEO;

            if (st->codec->codec_id == CODEC_ID_NONE) st->codec->codec_id = codec_get_bmp_id(st->codec->codec_tag);



            st->codec->width = s[i].width;

            st->codec->height = s[i].height;

            st->codec->sample_aspect_ratio.num = s[i].sample_width;

            st->codec->sample_aspect_ratio.den = s[i].sample_height;

            break;

        }

        if (st->codec->codec_id == CODEC_ID_NONE) av_log(avf, AV_LOG_ERROR, "Unknown codec?!\n");

    }



    return 0;

}
