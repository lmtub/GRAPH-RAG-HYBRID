static int mov_read_header(AVFormatContext *s)

{

    MOVContext *mov = s->priv_data;

    AVIOContext *pb = s->pb;

    int j, err;

    MOVAtom atom = { AV_RL32("root") };

    int i;



    if (mov->decryption_key_len != 0 && mov->decryption_key_len != AES_CTR_KEY_SIZE) {

        av_log(s, AV_LOG_ERROR, "Invalid decryption key len %d expected %d\n",

            mov->decryption_key_len, AES_CTR_KEY_SIZE);

        return AVERROR(EINVAL);

    }



    mov->fc = s;

    mov->trak_index = -1;

    /* .mov and .mp4 aren't streamable anyway (only progressive download if moov is before mdat) */

    if (pb->seekable)

        atom.size = avio_size(pb);

    else

        atom.size = INT64_MAX;



    /* check MOV header */

    do {

    if (mov->moov_retry)

        avio_seek(pb, 0, SEEK_SET);

    if ((err = mov_read_default(mov, pb, atom)) < 0) {

        av_log(s, AV_LOG_ERROR, "error reading header\n");

        mov_read_close(s);

        return err;

    }

    } while (pb->seekable && !mov->found_moov && !mov->moov_retry++);

    if (!mov->found_moov) {

        av_log(s, AV_LOG_ERROR, "moov atom not found\n");

        mov_read_close(s);

        return AVERROR_INVALIDDATA;

    }

    av_log(mov->fc, AV_LOG_TRACE, "on_parse_exit_offset=%"PRId64"\n", avio_tell(pb));



    if (pb->seekable) {

        if (mov->nb_chapter_tracks > 0 && !mov->ignore_chapters)

            mov_read_chapters(s);

        for (i = 0; i < s->nb_streams; i++)

            if (s->streams[i]->codecpar->codec_tag == AV_RL32("tmcd")) {

                mov_read_timecode_track(s, s->streams[i]);

            } else if (s->streams[i]->codecpar->codec_tag == AV_RL32("rtmd")) {

                mov_read_rtmd_track(s, s->streams[i]);

            }

    }



    /* copy timecode metadata from tmcd tracks to the related video streams */

    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MOVStreamContext *sc = st->priv_data;

        if (sc->timecode_track > 0) {

            AVDictionaryEntry *tcr;

            int tmcd_st_id = -1;



            for (j = 0; j < s->nb_streams; j++)

                if (s->streams[j]->id == sc->timecode_track)

                    tmcd_st_id = j;



            if (tmcd_st_id < 0 || tmcd_st_id == i)

                continue;

            tcr = av_dict_get(s->streams[tmcd_st_id]->metadata, "timecode", NULL, 0);

            if (tcr)

                av_dict_set(&st->metadata, "timecode", tcr->value, 0);

        }

    }

    export_orphan_timecode(s);



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MOVStreamContext *sc = st->priv_data;

        fix_timescale(mov, sc);

        if(st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && st->codecpar->codec_id == AV_CODEC_ID_AAC) {

            st->skip_samples = sc->start_pad;

        }

        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && sc->nb_frames_for_fps > 0 && sc->duration_for_fps > 0)

            av_reduce(&st->avg_frame_rate.num, &st->avg_frame_rate.den,

                      sc->time_scale*(int64_t)sc->nb_frames_for_fps, sc->duration_for_fps, INT_MAX);

        if (st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {

            if (st->codecpar->width <= 0 || st->codecpar->height <= 0) {

                st->codecpar->width  = sc->width;

                st->codecpar->height = sc->height;

            }

            if (st->codecpar->codec_id == AV_CODEC_ID_DVD_SUBTITLE) {

                if ((err = mov_rewrite_dvd_sub_extradata(st)) < 0)

                    return err;

            }

        }

        if (mov->handbrake_version &&

            mov->handbrake_version <= 1000000*0 + 1000*10 + 2 &&  // 0.10.2

            st->codecpar->codec_id == AV_CODEC_ID_MP3

        ) {

            av_log(s, AV_LOG_VERBOSE, "Forcing full parsing for mp3 stream\n");

            st->need_parsing = AVSTREAM_PARSE_FULL;

        }

    }



    if (mov->trex_data) {

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            MOVStreamContext *sc = st->priv_data;

            if (st->duration > 0)

                st->codecpar->bit_rate = sc->data_size * 8 * sc->time_scale / st->duration;

        }

    }



    if (mov->use_mfra_for > 0) {

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            MOVStreamContext *sc = st->priv_data;

            if (sc->duration_for_fps > 0) {

                st->codecpar->bit_rate = sc->data_size * 8 * sc->time_scale /

                    sc->duration_for_fps;

            }

        }

    }



    for (i = 0; i < mov->bitrates_count && i < s->nb_streams; i++) {

        if (mov->bitrates[i]) {

            s->streams[i]->codecpar->bit_rate = mov->bitrates[i];

        }

    }



    ff_rfps_calculate(s);



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MOVStreamContext *sc = st->priv_data;



        switch (st->codecpar->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            err = ff_replaygain_export(st, s->metadata);

            if (err < 0) {

                mov_read_close(s);

                return err;

            }

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (sc->display_matrix) {

                err = av_stream_add_side_data(st, AV_PKT_DATA_DISPLAYMATRIX, (uint8_t*)sc->display_matrix,

                                              sizeof(int32_t) * 9);

                if (err < 0)

                    return err;



                sc->display_matrix = NULL;

            }

            if (sc->stereo3d) {

                err = av_stream_add_side_data(st, AV_PKT_DATA_STEREO3D,

                                              (uint8_t *)sc->stereo3d,

                                              sizeof(*sc->stereo3d));

                if (err < 0)

                    return err;



                sc->stereo3d = NULL;

            }

            if (sc->spherical) {

                err = av_stream_add_side_data(st, AV_PKT_DATA_SPHERICAL,

                                              (uint8_t *)sc->spherical,

                                              sc->spherical_size);

                if (err < 0)

                    return err;



                sc->spherical = NULL;

            }

            break;

        }

    }

    ff_configure_buffers_for_index(s, AV_TIME_BASE);



    return 0;

}
