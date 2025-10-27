static int mov_flush_fragment(AVFormatContext *s)

{

    MOVMuxContext *mov = s->priv_data;

    int i, first_track = -1;

    int64_t mdat_size = 0;



    if (!(mov->flags & FF_MOV_FLAG_FRAGMENT))

        return 0;



    if (mov->fragments == 0) {

        int64_t pos = avio_tell(s->pb);

        uint8_t *buf;

        int buf_size, moov_size;



        for (i = 0; i < mov->nb_streams; i++)

            if (!mov->tracks[i].entry)

                break;

        /* Don't write the initial moov unless all tracks have data */

        if (i < mov->nb_streams)

            return 0;



        moov_size = get_moov_size(s);

        for (i = 0; i < mov->nb_streams; i++)

            mov->tracks[i].data_offset = pos + moov_size + 8;



        if (mov->flags & FF_MOV_FLAG_DELAY_MOOV)

            mov_write_identification(s->pb, s);

        mov_write_moov_tag(s->pb, mov, s);



        if (mov->flags & FF_MOV_FLAG_DELAY_MOOV) {

            if (mov->flags & FF_MOV_FLAG_FASTSTART)

                mov->reserved_moov_pos = avio_tell(s->pb);

            avio_flush(s->pb);

            mov->fragments++;

            return 0;

        }



        buf_size = avio_close_dyn_buf(mov->mdat_buf, &buf);

        mov->mdat_buf = NULL;

        avio_wb32(s->pb, buf_size + 8);

        ffio_wfourcc(s->pb, "mdat");

        avio_write(s->pb, buf, buf_size);

        av_free(buf);



        mov->fragments++;

        mov->mdat_size = 0;

        for (i = 0; i < mov->nb_streams; i++) {

            if (mov->tracks[i].entry)

                mov->tracks[i].frag_start += mov->tracks[i].start_dts +

                                             mov->tracks[i].track_duration -

                                             mov->tracks[i].cluster[0].dts;

            mov->tracks[i].entry = 0;

        }

        avio_flush(s->pb);

        return 0;

    }



    for (i = 0; i < mov->nb_streams; i++) {

        MOVTrack *track = &mov->tracks[i];

        if (mov->flags & FF_MOV_FLAG_SEPARATE_MOOF)

            track->data_offset = 0;

        else

            track->data_offset = mdat_size;

        if (!track->mdat_buf)

            continue;

        mdat_size += avio_tell(track->mdat_buf);

        if (first_track < 0)

            first_track = i;

    }



    if (!mdat_size)

        return 0;



    for (i = 0; i < mov->nb_streams; i++) {

        MOVTrack *track = &mov->tracks[i];

        int buf_size, write_moof = 1, moof_tracks = -1;

        uint8_t *buf;

        int64_t duration = 0;



        if (track->entry)

            duration = track->start_dts + track->track_duration -

                       track->cluster[0].dts;

        if (mov->flags & FF_MOV_FLAG_SEPARATE_MOOF) {

            if (!track->mdat_buf)

                continue;

            mdat_size = avio_tell(track->mdat_buf);

            moof_tracks = i;

        } else {

            write_moof = i == first_track;

        }



        if (write_moof) {

            avio_flush(s->pb);



            mov_write_moof_tag(s->pb, mov, moof_tracks, mdat_size);

            mov->fragments++;



            avio_wb32(s->pb, mdat_size + 8);

            ffio_wfourcc(s->pb, "mdat");

        }



        if (track->entry)

            track->frag_start += duration;

        track->entry = 0;

        if (!track->mdat_buf)

            continue;

        buf_size = avio_close_dyn_buf(track->mdat_buf, &buf);

        track->mdat_buf = NULL;



        avio_write(s->pb, buf, buf_size);

        av_free(buf);

    }



    mov->mdat_size = 0;



    avio_flush(s->pb);

    return 0;

}
