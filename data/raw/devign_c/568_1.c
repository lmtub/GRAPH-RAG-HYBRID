static int mkv_write_header(AVFormatContext *s)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVIOContext *pb = s->pb;

    ebml_master ebml_header;

    AVDictionaryEntry *tag;

    int ret, i, version = 2;

    int64_t creation_time;



    if (!strcmp(s->oformat->name, "webm"))

        mkv->mode = MODE_WEBM;

    else

        mkv->mode = MODE_MATROSKAv2;



    if (mkv->mode != MODE_WEBM ||

        av_dict_get(s->metadata, "stereo_mode", NULL, 0) ||

        av_dict_get(s->metadata, "alpha_mode", NULL, 0))

        version = 4;



    for (i = 0; i < s->nb_streams; i++) {

        if (s->streams[i]->codecpar->codec_id == AV_CODEC_ID_ATRAC3 ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_COOK ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_RA_288 ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_SIPR ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_RV10 ||

            s->streams[i]->codecpar->codec_id == AV_CODEC_ID_RV20) {

            av_log(s, AV_LOG_ERROR,

                   "The Matroska muxer does not yet support muxing %s\n",

                   avcodec_get_name(s->streams[i]->codecpar->codec_id));

            return AVERROR_PATCHWELCOME;

        }

        if (s->streams[i]->codecpar->codec_id == AV_CODEC_ID_OPUS ||

            av_dict_get(s->streams[i]->metadata, "stereo_mode", NULL, 0) ||

            av_dict_get(s->streams[i]->metadata, "alpha_mode", NULL, 0))

            version = 4;

    }



    mkv->tracks = av_mallocz_array(s->nb_streams, sizeof(*mkv->tracks));

    if (!mkv->tracks) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    ebml_header = start_ebml_master(pb, EBML_ID_HEADER, 0);

    put_ebml_uint   (pb, EBML_ID_EBMLVERSION        ,           1);

    put_ebml_uint   (pb, EBML_ID_EBMLREADVERSION    ,           1);

    put_ebml_uint   (pb, EBML_ID_EBMLMAXIDLENGTH    ,           4);

    put_ebml_uint   (pb, EBML_ID_EBMLMAXSIZELENGTH  ,           8);

    put_ebml_string (pb, EBML_ID_DOCTYPE            , s->oformat->name);

    put_ebml_uint   (pb, EBML_ID_DOCTYPEVERSION     ,     version);

    put_ebml_uint   (pb, EBML_ID_DOCTYPEREADVERSION ,           2);

    end_ebml_master(pb, ebml_header);



    mkv->segment = start_ebml_master(pb, MATROSKA_ID_SEGMENT, 0);

    mkv->segment_offset = avio_tell(pb);



    // we write 2 seek heads - one at the end of the file to point to each

    // cluster, and one at the beginning to point to all other level one

    // elements (including the seek head at the end of the file), which

    // isn't more than 10 elements if we only write one of each other

    // currently defined level 1 element

    mkv->main_seekhead    = mkv_start_seekhead(pb, mkv->segment_offset, 10);

    if (!mkv->main_seekhead) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    ret = mkv_add_seekhead_entry(mkv->main_seekhead, MATROSKA_ID_INFO, avio_tell(pb));

    if (ret < 0) goto fail;



    ret = start_ebml_master_crc32(pb, &mkv->info_bc, mkv, &mkv->info, MATROSKA_ID_INFO, 0);

    if (ret < 0)

        return ret;

    pb = mkv->info_bc;



    put_ebml_uint(pb, MATROSKA_ID_TIMECODESCALE, 1000000);

    if ((tag = av_dict_get(s->metadata, "title", NULL, 0)))

        put_ebml_string(pb, MATROSKA_ID_TITLE, tag->value);

    if (!(s->flags & AVFMT_FLAG_BITEXACT)) {

        put_ebml_string(pb, MATROSKA_ID_MUXINGAPP, LIBAVFORMAT_IDENT);

        if ((tag = av_dict_get(s->metadata, "encoding_tool", NULL, 0)))

            put_ebml_string(pb, MATROSKA_ID_WRITINGAPP, tag->value);

        else

            put_ebml_string(pb, MATROSKA_ID_WRITINGAPP, LIBAVFORMAT_IDENT);



        if (mkv->mode != MODE_WEBM) {

            uint32_t segment_uid[4];

            AVLFG lfg;



            av_lfg_init(&lfg, av_get_random_seed());



            for (i = 0; i < 4; i++)

                segment_uid[i] = av_lfg_get(&lfg);



            put_ebml_binary(pb, MATROSKA_ID_SEGMENTUID, segment_uid, 16);

        }

    } else {

        const char *ident = "Lavf";

        put_ebml_string(pb, MATROSKA_ID_MUXINGAPP , ident);

        put_ebml_string(pb, MATROSKA_ID_WRITINGAPP, ident);

    }



    if (ff_parse_creation_time_metadata(s, &creation_time, 0) > 0) {

        // Adjust time so it's relative to 2001-01-01 and convert to nanoseconds.

        int64_t date_utc = (creation_time - 978307200000000LL) * 1000;

        uint8_t date_utc_buf[8];

        AV_WB64(date_utc_buf, date_utc);

        put_ebml_binary(pb, MATROSKA_ID_DATEUTC, date_utc_buf, 8);

    }



    // reserve space for the duration

    mkv->duration = 0;

    mkv->duration_offset = avio_tell(pb);

    if (!mkv->is_live) {

        int64_t metadata_duration = get_metadata_duration(s);



        if (s->duration > 0) {

            int64_t scaledDuration = av_rescale(s->duration, 1000, AV_TIME_BASE);

            put_ebml_float(pb, MATROSKA_ID_DURATION, scaledDuration);

            av_log(s, AV_LOG_DEBUG, "Write early duration from recording time = %" PRIu64 "\n", scaledDuration);

        } else if (metadata_duration > 0) {

            int64_t scaledDuration = av_rescale(metadata_duration, 1000, AV_TIME_BASE);

            put_ebml_float(pb, MATROSKA_ID_DURATION, scaledDuration);

            av_log(s, AV_LOG_DEBUG, "Write early duration from metadata = %" PRIu64 "\n", scaledDuration);

        } else {

            put_ebml_void(pb, 11);              // assumes double-precision float to be written

        }

    }

    if (s->pb->seekable && !mkv->is_live)

        put_ebml_void(s->pb, avio_tell(pb));

    else

        end_ebml_master_crc32(s->pb, &mkv->info_bc, mkv, mkv->info);

    pb = s->pb;



    // initialize stream_duration fields

    mkv->stream_durations = av_mallocz(s->nb_streams * sizeof(int64_t));

    mkv->stream_duration_offsets = av_mallocz(s->nb_streams * sizeof(int64_t));



    ret = mkv_write_tracks(s);

    if (ret < 0)

        goto fail;



    for (i = 0; i < s->nb_chapters; i++)

        mkv->chapter_id_offset = FFMAX(mkv->chapter_id_offset, 1LL - s->chapters[i]->id);



    if (mkv->mode != MODE_WEBM) {

        ret = mkv_write_chapters(s);

        if (ret < 0)

            goto fail;



        ret = mkv_write_attachments(s);

        if (ret < 0)

            goto fail;



        ret = mkv_write_tags(s);

        if (ret < 0)

            goto fail;

    }



    if (!s->pb->seekable && !mkv->is_live)

        mkv_write_seekhead(pb, mkv);



    mkv->cues = mkv_start_cues(mkv->segment_offset);

    if (!mkv->cues) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    if (pb->seekable && mkv->reserve_cues_space) {

        mkv->cues_pos = avio_tell(pb);

        put_ebml_void(pb, mkv->reserve_cues_space);

    }



    av_init_packet(&mkv->cur_audio_pkt);

    mkv->cur_audio_pkt.size = 0;

    mkv->cluster_pos = -1;



    avio_flush(pb);



    // start a new cluster every 5 MB or 5 sec, or 32k / 1 sec for streaming or

    // after 4k and on a keyframe

    if (pb->seekable) {

        if (mkv->cluster_time_limit < 0)

            mkv->cluster_time_limit = 5000;

        if (mkv->cluster_size_limit < 0)

            mkv->cluster_size_limit = 5 * 1024 * 1024;

    } else {

        if (mkv->cluster_time_limit < 0)

            mkv->cluster_time_limit = 1000;

        if (mkv->cluster_size_limit < 0)

            mkv->cluster_size_limit = 32 * 1024;

    }



    return 0;

fail:

    mkv_free(mkv);

    return ret;

}
