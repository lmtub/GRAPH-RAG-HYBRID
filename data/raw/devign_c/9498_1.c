static int matroska_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    MatroskaDemuxContext *matroska = s->priv_data;

    EbmlList *attachements_list = &matroska->attachments;

    MatroskaAttachement *attachements;

    EbmlList *chapters_list = &matroska->chapters;

    MatroskaChapter *chapters;

    MatroskaTrack *tracks;

    EbmlList *index_list;

    MatroskaIndex *index;

    int index_scale = 1;

    Ebml ebml = { 0 };

    AVStream *st;

    int i, j;



    matroska->ctx = s;



    /* First read the EBML header. */

    if (ebml_parse(matroska, ebml_syntax, &ebml)

        || ebml.version > EBML_VERSION       || ebml.max_size > sizeof(uint64_t)

        || ebml.id_length > sizeof(uint32_t) || strcmp(ebml.doctype, "matroska")

        || ebml.doctype_version > 2) {

        av_log(matroska->ctx, AV_LOG_ERROR,

               "EBML header using unsupported features\n"

               "(EBML version %"PRIu64", doctype %s, doc version %"PRIu64")\n",

               ebml.version, ebml.doctype, ebml.doctype_version);

        return AVERROR_NOFMT;

    }

    ebml_free(ebml_syntax, &ebml);



    /* The next thing is a segment. */

    if (ebml_parse(matroska, matroska_segments, matroska) < 0)

        return -1;

    matroska_execute_seekhead(matroska);



    if (matroska->duration)

        matroska->ctx->duration = matroska->duration * matroska->time_scale

                                  * 1000 / AV_TIME_BASE;

    if (matroska->title)

        strncpy(matroska->ctx->title, matroska->title,

                sizeof(matroska->ctx->title)-1);

    matroska_convert_tags(s, &matroska->tags);



    tracks = matroska->tracks.elem;

    for (i=0; i < matroska->tracks.nb_elem; i++) {

        MatroskaTrack *track = &tracks[i];

        enum CodecID codec_id = CODEC_ID_NONE;

        EbmlList *encodings_list = &tracks->encodings;

        MatroskaTrackEncoding *encodings = encodings_list->elem;

        uint8_t *extradata = NULL;

        int extradata_size = 0;

        int extradata_offset = 0;



        /* Apply some sanity checks. */

        if (track->type != MATROSKA_TRACK_TYPE_VIDEO &&

            track->type != MATROSKA_TRACK_TYPE_AUDIO &&

            track->type != MATROSKA_TRACK_TYPE_SUBTITLE) {

            av_log(matroska->ctx, AV_LOG_INFO,

                   "Unknown or unsupported track type %"PRIu64"\n",

                   track->type);

            continue;

        }

        if (track->codec_id == NULL)

            continue;



        if (track->type == MATROSKA_TRACK_TYPE_VIDEO) {

            if (!track->default_duration)

                track->default_duration = 1000000000/track->video.frame_rate;

            if (!track->video.display_width)

                track->video.display_width = track->video.pixel_width;

            if (!track->video.display_height)

                track->video.display_height = track->video.pixel_height;

        } else if (track->type == MATROSKA_TRACK_TYPE_AUDIO) {

            if (!track->audio.out_samplerate)

                track->audio.out_samplerate = track->audio.samplerate;

        }

        if (encodings_list->nb_elem > 1) {

            av_log(matroska->ctx, AV_LOG_ERROR,

                   "Multiple combined encodings no supported");

        } else if (encodings_list->nb_elem == 1) {

            if (encodings[0].type ||

                (encodings[0].compression.algo != MATROSKA_TRACK_ENCODING_COMP_HEADERSTRIP &&

#ifdef CONFIG_ZLIB

                 encodings[0].compression.algo != MATROSKA_TRACK_ENCODING_COMP_ZLIB &&

#endif

#ifdef CONFIG_BZLIB

                 encodings[0].compression.algo != MATROSKA_TRACK_ENCODING_COMP_BZLIB &&

#endif

                 encodings[0].compression.algo != MATROSKA_TRACK_ENCODING_COMP_LZO)) {

                encodings[0].scope = 0;

                av_log(matroska->ctx, AV_LOG_ERROR,

                       "Unsupported encoding type");

            } else if (track->codec_priv.size && encodings[0].scope&2) {

                uint8_t *codec_priv = track->codec_priv.data;

                int offset = matroska_decode_buffer(&track->codec_priv.data,

                                                    &track->codec_priv.size,

                                                    track);

                if (offset < 0) {

                    track->codec_priv.data = NULL;

                    track->codec_priv.size = 0;

                    av_log(matroska->ctx, AV_LOG_ERROR,

                           "Failed to decode codec private data\n");

                } else if (offset > 0) {

                    track->codec_priv.data = av_malloc(track->codec_priv.size + offset);

                    memcpy(track->codec_priv.data,

                           encodings[0].compression.settings.data, offset);

                    memcpy(track->codec_priv.data+offset, codec_priv,

                           track->codec_priv.size);

                    track->codec_priv.size += offset;

                }

                if (codec_priv != track->codec_priv.data)

                    av_free(codec_priv);

            }

        }



        for(j=0; ff_mkv_codec_tags[j].id != CODEC_ID_NONE; j++){

            if(!strncmp(ff_mkv_codec_tags[j].str, track->codec_id,

                        strlen(ff_mkv_codec_tags[j].str))){

                codec_id= ff_mkv_codec_tags[j].id;

                break;

            }

        }



        st = track->stream = av_new_stream(s, 0);

        if (st == NULL)

            return AVERROR(ENOMEM);



        if (!strcmp(track->codec_id, "V_MS/VFW/FOURCC")

            && track->codec_priv.size >= 40

            && track->codec_priv.data != NULL) {

            track->video.fourcc = AV_RL32(track->codec_priv.data + 16);

            codec_id = codec_get_id(codec_bmp_tags, track->video.fourcc);

        } else if (!strcmp(track->codec_id, "A_MS/ACM")

                   && track->codec_priv.size >= 18

                   && track->codec_priv.data != NULL) {

            uint16_t tag = AV_RL16(track->codec_priv.data);

            codec_id = codec_get_id(codec_wav_tags, tag);

        } else if (!strcmp(track->codec_id, "V_QUICKTIME")

                   && (track->codec_priv.size >= 86)

                   && (track->codec_priv.data != NULL)) {

            track->video.fourcc = AV_RL32(track->codec_priv.data);

            codec_id=codec_get_id(codec_movvideo_tags, track->video.fourcc);

        } else if (codec_id == CODEC_ID_PCM_S16BE) {

            switch (track->audio.bitdepth) {

            case  8:  codec_id = CODEC_ID_PCM_U8;     break;

            case 24:  codec_id = CODEC_ID_PCM_S24BE;  break;

            case 32:  codec_id = CODEC_ID_PCM_S32BE;  break;

            }

        } else if (codec_id == CODEC_ID_PCM_S16LE) {

            switch (track->audio.bitdepth) {

            case  8:  codec_id = CODEC_ID_PCM_U8;     break;

            case 24:  codec_id = CODEC_ID_PCM_S24LE;  break;

            case 32:  codec_id = CODEC_ID_PCM_S32LE;  break;

            }

        } else if (codec_id==CODEC_ID_PCM_F32LE && track->audio.bitdepth==64) {

            codec_id = CODEC_ID_PCM_F64LE;

        } else if (codec_id == CODEC_ID_AAC && !track->codec_priv.size) {

            int profile = matroska_aac_profile(track->codec_id);

            int sri = matroska_aac_sri(track->audio.samplerate);

            extradata = av_malloc(5);

            if (extradata == NULL)

                return AVERROR(ENOMEM);

            extradata[0] = (profile << 3) | ((sri&0x0E) >> 1);

            extradata[1] = ((sri&0x01) << 7) | (track->audio.channels<<3);

            if (strstr(track->codec_id, "SBR")) {

                sri = matroska_aac_sri(track->audio.out_samplerate);

                extradata[2] = 0x56;

                extradata[3] = 0xE5;

                extradata[4] = 0x80 | (sri<<3);

                extradata_size = 5;

            } else

                extradata_size = 2;

        } else if (codec_id == CODEC_ID_TTA) {

            ByteIOContext b;

            extradata_size = 30;

            extradata = av_mallocz(extradata_size);

            if (extradata == NULL)

                return AVERROR(ENOMEM);

            init_put_byte(&b, extradata, extradata_size, 1,

                          NULL, NULL, NULL, NULL);

            put_buffer(&b, "TTA1", 4);

            put_le16(&b, 1);

            put_le16(&b, track->audio.channels);

            put_le16(&b, track->audio.bitdepth);

            put_le32(&b, track->audio.out_samplerate);

            put_le32(&b, matroska->ctx->duration * track->audio.out_samplerate);

        } else if (codec_id == CODEC_ID_RV10 || codec_id == CODEC_ID_RV20 ||

                   codec_id == CODEC_ID_RV30 || codec_id == CODEC_ID_RV40) {

            extradata_offset = 26;

            track->codec_priv.size -= extradata_offset;

        } else if (codec_id == CODEC_ID_RA_144) {

            track->audio.out_samplerate = 8000;

            track->audio.channels = 1;

        } else if (codec_id == CODEC_ID_RA_288 || codec_id == CODEC_ID_COOK ||

                   codec_id == CODEC_ID_ATRAC3) {

            ByteIOContext b;



            init_put_byte(&b, track->codec_priv.data,track->codec_priv.size,

                          0, NULL, NULL, NULL, NULL);

            url_fskip(&b, 24);

            track->audio.coded_framesize = get_be32(&b);

            url_fskip(&b, 12);

            track->audio.sub_packet_h    = get_be16(&b);

            track->audio.frame_size      = get_be16(&b);

            track->audio.sub_packet_size = get_be16(&b);

            track->audio.buf = av_malloc(track->audio.frame_size * track->audio.sub_packet_h);

            if (codec_id == CODEC_ID_RA_288) {

                st->codec->block_align = track->audio.coded_framesize;

                track->codec_priv.size = 0;

            } else {

                st->codec->block_align = track->audio.sub_packet_size;

                extradata_offset = 78;

                track->codec_priv.size -= extradata_offset;

            }

        }



        if (codec_id == CODEC_ID_NONE)

            av_log(matroska->ctx, AV_LOG_INFO,

                   "Unknown/unsupported CodecID %s.\n", track->codec_id);



        if (track->time_scale < 0.01)

            track->time_scale = 1.0;

        av_set_pts_info(st, 64, matroska->time_scale*track->time_scale, 1000*1000*1000); /* 64 bit pts in ns */



        st->codec->codec_id = codec_id;

        st->start_time = 0;

        if (strcmp(track->language, "und"))

            av_strlcpy(st->language, track->language, 4);



        if (track->flag_default)

            st->disposition |= AV_DISPOSITION_DEFAULT;



        if (track->default_duration)

            av_reduce(&st->codec->time_base.num, &st->codec->time_base.den,

                      track->default_duration, 1000000000, 30000);



        if(extradata){

            st->codec->extradata = extradata;

            st->codec->extradata_size = extradata_size;

        } else if(track->codec_priv.data && track->codec_priv.size > 0){

            st->codec->extradata = av_malloc(track->codec_priv.size);

            if(st->codec->extradata == NULL)

                return AVERROR(ENOMEM);

            st->codec->extradata_size = track->codec_priv.size;

            memcpy(st->codec->extradata,

                   track->codec_priv.data + extradata_offset,

                   track->codec_priv.size);

        }



        if (track->type == MATROSKA_TRACK_TYPE_VIDEO) {

            st->codec->codec_type = CODEC_TYPE_VIDEO;

            st->codec->codec_tag  = track->video.fourcc;

            st->codec->width  = track->video.pixel_width;

            st->codec->height = track->video.pixel_height;

            av_reduce(&st->sample_aspect_ratio.num,

                      &st->sample_aspect_ratio.den,

                      st->codec->height * track->video.display_width,

                      st->codec-> width * track->video.display_height,

                      255);

            st->need_parsing = AVSTREAM_PARSE_HEADERS;

        } else if (track->type == MATROSKA_TRACK_TYPE_AUDIO) {

            st->codec->codec_type = CODEC_TYPE_AUDIO;

            st->codec->sample_rate = track->audio.out_samplerate;

            st->codec->channels = track->audio.channels;

        } else if (track->type == MATROSKA_TRACK_TYPE_SUBTITLE) {

            st->codec->codec_type = CODEC_TYPE_SUBTITLE;

        }

    }



    attachements = attachements_list->elem;

    for (j=0; j<attachements_list->nb_elem; j++) {

        if (!(attachements[j].filename && attachements[j].mime &&

              attachements[j].bin.data && attachements[j].bin.size > 0)) {

            av_log(matroska->ctx, AV_LOG_ERROR, "incomplete attachment\n");

        } else {

            AVStream *st = av_new_stream(s, 0);

            if (st == NULL)

                break;

            st->filename          = av_strdup(attachements[j].filename);

            st->codec->codec_id = CODEC_ID_NONE;

            st->codec->codec_type = CODEC_TYPE_ATTACHMENT;

            st->codec->extradata  = av_malloc(attachements[j].bin.size);

            if(st->codec->extradata == NULL)

                break;

            st->codec->extradata_size = attachements[j].bin.size;

            memcpy(st->codec->extradata, attachements[j].bin.data, attachements[j].bin.size);



            for (i=0; ff_mkv_mime_tags[i].id != CODEC_ID_NONE; i++) {

                if (!strncmp(ff_mkv_mime_tags[i].str, attachements[j].mime,

                             strlen(ff_mkv_mime_tags[i].str))) {

                    st->codec->codec_id = ff_mkv_mime_tags[i].id;

                    break;

                }

            }

        }

    }



    chapters = chapters_list->elem;

    for (i=0; i<chapters_list->nb_elem; i++)

        if (chapters[i].start != AV_NOPTS_VALUE && chapters[i].uid)

            ff_new_chapter(s, chapters[i].uid, (AVRational){1, 1000000000},

                           chapters[i].start, chapters[i].end,

                           chapters[i].title);



    index_list = &matroska->index;

    index = index_list->elem;

    if (index_list->nb_elem

        && index[0].time > 100000000000000/matroska->time_scale) {

        av_log(matroska->ctx, AV_LOG_WARNING, "Working around broken index.\n");

        index_scale = matroska->time_scale;

    }

    for (i=0; i<index_list->nb_elem; i++) {

        EbmlList *pos_list = &index[i].pos;

        MatroskaIndexPos *pos = pos_list->elem;

        for (j=0; j<pos_list->nb_elem; j++) {

            MatroskaTrack *track = matroska_find_track_by_num(matroska,

                                                              pos[j].track);

            if (track && track->stream)

                av_add_index_entry(track->stream,

                                   pos[j].pos + matroska->segment_start,

                                   index[i].time/index_scale, 0, 0,

                                   AVINDEX_KEYFRAME);

        }

    }



    return 0;

}
