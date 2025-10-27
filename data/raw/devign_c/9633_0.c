static int mkv_write_attachments(AVFormatContext *s)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVIOContext *dyn_cp, *pb = s->pb;

    ebml_master attachments;

    AVLFG c;

    int i, ret;



    if (!mkv->have_attachments)

        return 0;



    mkv->attachments = av_mallocz(sizeof(*mkv->attachments));

    if (!mkv->attachments)

        return ret;



    av_lfg_init(&c, av_get_random_seed());



    ret = mkv_add_seekhead_entry(mkv->main_seekhead, MATROSKA_ID_ATTACHMENTS, avio_tell(pb));

    if (ret < 0) return ret;



    ret = start_ebml_master_crc32(pb, &dyn_cp, &attachments, MATROSKA_ID_ATTACHMENTS, 0);

    if (ret < 0) return ret;



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        ebml_master attached_file;

        mkv_attachment *attachment = mkv->attachments->entries;

        AVDictionaryEntry *t;

        const char *mimetype = NULL;

        uint32_t fileuid;



        if (st->codecpar->codec_type != AVMEDIA_TYPE_ATTACHMENT)

            continue;



        attachment = av_realloc_array(attachment, mkv->attachments->num_entries + 1, sizeof(mkv_attachment));

        if (!attachment)

            return AVERROR(ENOMEM);

        mkv->attachments->entries = attachment;



        attached_file = start_ebml_master(dyn_cp, MATROSKA_ID_ATTACHEDFILE, 0);



        if (t = av_dict_get(st->metadata, "title", NULL, 0))

            put_ebml_string(dyn_cp, MATROSKA_ID_FILEDESC, t->value);

        if (!(t = av_dict_get(st->metadata, "filename", NULL, 0))) {

            av_log(s, AV_LOG_ERROR, "Attachment stream %d has no filename tag.\n", i);

            return AVERROR(EINVAL);

        }

        put_ebml_string(dyn_cp, MATROSKA_ID_FILENAME, t->value);

        if (t = av_dict_get(st->metadata, "mimetype", NULL, 0))

            mimetype = t->value;

        else if (st->codecpar->codec_id != AV_CODEC_ID_NONE ) {

            int i;

            for (i = 0; ff_mkv_mime_tags[i].id != AV_CODEC_ID_NONE; i++)

                if (ff_mkv_mime_tags[i].id == st->codecpar->codec_id) {

                    mimetype = ff_mkv_mime_tags[i].str;

                    break;

                }

            for (i = 0; ff_mkv_image_mime_tags[i].id != AV_CODEC_ID_NONE; i++)

                if (ff_mkv_image_mime_tags[i].id == st->codecpar->codec_id) {

                    mimetype = ff_mkv_image_mime_tags[i].str;

                    break;

                }

        }

        if (!mimetype) {

            av_log(s, AV_LOG_ERROR, "Attachment stream %d has no mimetype tag and "

                                    "it cannot be deduced from the codec id.\n", i);

            return AVERROR(EINVAL);

        }



        if (s->flags & AVFMT_FLAG_BITEXACT) {

            struct AVSHA *sha = av_sha_alloc();

            uint8_t digest[20];

            if (!sha)

                return AVERROR(ENOMEM);

            av_sha_init(sha, 160);

            av_sha_update(sha, st->codecpar->extradata, st->codecpar->extradata_size);

            av_sha_final(sha, digest);

            av_free(sha);

            fileuid = AV_RL32(digest);

        } else {

            fileuid = av_lfg_get(&c);

        }

        av_log(s, AV_LOG_VERBOSE, "Using %.8"PRIx32" for attachment %d\n",

               fileuid, mkv->attachments->num_entries);



        put_ebml_string(dyn_cp, MATROSKA_ID_FILEMIMETYPE, mimetype);

        put_ebml_binary(dyn_cp, MATROSKA_ID_FILEDATA, st->codecpar->extradata, st->codecpar->extradata_size);

        put_ebml_uint(dyn_cp, MATROSKA_ID_FILEUID, fileuid);

        end_ebml_master(dyn_cp, attached_file);



        mkv->attachments->entries[mkv->attachments->num_entries].stream_idx = i;

        mkv->attachments->entries[mkv->attachments->num_entries++].fileuid  = fileuid;

    }

    end_ebml_master_crc32(pb, &dyn_cp, mkv, attachments);



    return 0;

}
