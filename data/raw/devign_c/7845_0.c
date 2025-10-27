static int mov_read_header(AVFormatContext *s)

{

    MOVContext *mov = s->priv_data;

    AVIOContext *pb = s->pb;

    int err;

    MOVAtom atom = { AV_RL32("root") };

    int i;



    mov->fc = s;

    /* .mov and .mp4 aren't streamable anyway (only progressive download if moov is before mdat) */

    if (pb->seekable)

        atom.size = avio_size(pb);

    else

        atom.size = INT64_MAX;



    /* check MOV header */

    if ((err = mov_read_default(mov, pb, atom)) < 0) {

        av_log(s, AV_LOG_ERROR, "error reading header: %d\n", err);

        mov_read_close(s);

        return err;

    }

    if (!mov->found_moov) {

        av_log(s, AV_LOG_ERROR, "moov atom not found\n");

        mov_read_close(s);

        return AVERROR_INVALIDDATA;

    }

    av_dlog(mov->fc, "on_parse_exit_offset=%"PRId64"\n", avio_tell(pb));



    if (pb->seekable && mov->chapter_track > 0)

        mov_read_chapters(s);



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MOVStreamContext *sc = st->priv_data;



        if (st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) {

            if (st->codec->width <= 0 && st->codec->width <= 0) {

                st->codec->width  = sc->width;

                st->codec->height = sc->height;

            }

            if (st->codec->codec_id == AV_CODEC_ID_DVD_SUBTITLE) {

                if ((err = mov_rewrite_dvd_sub_extradata(st)) < 0)

                    return err;

            }

        }

    }



    if (mov->trex_data) {

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            MOVStreamContext *sc = st->priv_data;

            if (st->duration > 0)

                st->codec->bit_rate = sc->data_size * 8 * sc->time_scale / st->duration;

        }

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MOVStreamContext *sc = st->priv_data;



        switch (st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            err = ff_replaygain_export(st, s->metadata);

            if (err < 0) {

                mov_read_close(s);

                return err;

            }

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (sc->display_matrix) {

                AVPacketSideData *sd, *tmp;



                tmp = av_realloc_array(st->side_data,

                                       st->nb_side_data + 1, sizeof(*tmp));

                if (!tmp)

                    return AVERROR(ENOMEM);



                st->side_data = tmp;

                st->nb_side_data++;



                sd = &st->side_data[st->nb_side_data - 1];

                sd->type = AV_PKT_DATA_DISPLAYMATRIX;

                sd->size = sizeof(int32_t) * 9;

                sd->data = (uint8_t*)sc->display_matrix;

                sc->display_matrix = NULL;

            }

            break;

        }

    }



    return 0;

}
