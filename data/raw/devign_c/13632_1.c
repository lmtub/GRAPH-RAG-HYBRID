static int decode_info_header(NUTContext *nut)

{

    AVFormatContext *s = nut->avf;

    AVIOContext *bc    = s->pb;

    uint64_t tmp, chapter_start, chapter_len;

    unsigned int stream_id_plus1, count;

    int chapter_id, i;

    int64_t value, end;

    char name[256], str_value[1024], type_str[256];

    const char *type;

    int *event_flags;

    AVChapter *chapter      = NULL;

    AVStream *st            = NULL;

    AVDictionary **metadata = NULL;

    int metadata_flag       = 0;



    end  = get_packetheader(nut, bc, 1, INFO_STARTCODE);

    end += avio_tell(bc);



    GET_V(stream_id_plus1, tmp <= s->nb_streams);

    chapter_id    = get_s(bc);

    chapter_start = ffio_read_varlen(bc);

    chapter_len   = ffio_read_varlen(bc);

    count         = ffio_read_varlen(bc);



    if (chapter_id && !stream_id_plus1) {

        int64_t start = chapter_start / nut->time_base_count;

        chapter = avpriv_new_chapter(s, chapter_id,

                                     nut->time_base[chapter_start %

                                                    nut->time_base_count],

                                     start, start + chapter_len, NULL);

        metadata = &chapter->metadata;

    } else if (stream_id_plus1) {

        st       = s->streams[stream_id_plus1 - 1];

        metadata = &st->metadata;

        event_flags = &st->event_flags;

        metadata_flag = AVSTREAM_EVENT_FLAG_METADATA_UPDATED;

    } else {

        metadata = &s->metadata;

        event_flags = &s->event_flags;

        metadata_flag = AVFMT_EVENT_FLAG_METADATA_UPDATED;

    }



    for (i = 0; i < count; i++) {

        get_str(bc, name, sizeof(name));

        value = get_s(bc);

        if (value == -1) {

            type = "UTF-8";

            get_str(bc, str_value, sizeof(str_value));

        } else if (value == -2) {

            get_str(bc, type_str, sizeof(type_str));

            type = type_str;

            get_str(bc, str_value, sizeof(str_value));

        } else if (value == -3) {

            type  = "s";

            value = get_s(bc);

        } else if (value == -4) {

            type  = "t";

            value = ffio_read_varlen(bc);

        } else if (value < -4) {

            type = "r";

            get_s(bc);

        } else {

            type = "v";

        }



        if (stream_id_plus1 > s->nb_streams) {

            av_log(s, AV_LOG_ERROR, "invalid stream id for info packet\n");

            continue;

        }



        if (!strcmp(type, "UTF-8")) {

            if (chapter_id == 0 && !strcmp(name, "Disposition")) {

                set_disposition_bits(s, str_value, stream_id_plus1 - 1);

                continue;

            }

            if (metadata && av_strcasecmp(name, "Uses") &&

                av_strcasecmp(name, "Depends") && av_strcasecmp(name, "Replaces")) {

                *event_flags |= metadata_flag;

                av_dict_set(metadata, name, str_value, 0);

            }

        }

    }



    if (skip_reserved(bc, end) || ffio_get_checksum(bc)) {

        av_log(s, AV_LOG_ERROR, "info header checksum mismatch\n");

        return AVERROR_INVALIDDATA;

    }

    return 0;

}
