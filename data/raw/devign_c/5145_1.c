static int mkv_write_tags(AVFormatContext *s)

{

    MatroskaMuxContext *mkv = s->priv_data;

    int i, ret;



    ff_metadata_conv_ctx(s, ff_mkv_metadata_conv, NULL);



    if (mkv_check_tag(s->metadata, 0)) {

        ret = mkv_write_tag(s, s->metadata, 0, 0, &mkv->tags);

        if (ret < 0) return ret;

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];



        if (st->codecpar->codec_type == AVMEDIA_TYPE_ATTACHMENT)

            continue;



        if (!mkv_check_tag(st->metadata, MATROSKA_ID_TAGTARGETS_TRACKUID))

            continue;



        ret = mkv_write_tag(s, st->metadata, MATROSKA_ID_TAGTARGETS_TRACKUID, i + 1, &mkv->tags);

        if (ret < 0) return ret;

    }



    if (s->pb->seekable && !mkv->is_live) {

        for (i = 0; i < s->nb_streams; i++) {

            AVIOContext *pb;

            AVStream *st = s->streams[i];

            ebml_master tag_target;

            ebml_master tag;



            if (st->codecpar->codec_type == AVMEDIA_TYPE_ATTACHMENT)

                continue;



            mkv_write_tag_targets(s, MATROSKA_ID_TAGTARGETS_TRACKUID, i + 1, &mkv->tags, &tag_target);

            pb = mkv->tags_bc;



            tag = start_ebml_master(pb, MATROSKA_ID_SIMPLETAG, 0);

            put_ebml_string(pb, MATROSKA_ID_TAGNAME, "DURATION");

            mkv->stream_duration_offsets[i] = avio_tell(pb);



            // Reserve space to write duration as a 20-byte string.

            // 2 (ebml id) + 1 (data size) + 20 (data)

            put_ebml_void(pb, 23);

            end_ebml_master(pb, tag);

            end_ebml_master(pb, tag_target);

        }

    }



    for (i = 0; i < s->nb_chapters; i++) {

        AVChapter *ch = s->chapters[i];



        if (!mkv_check_tag(ch->metadata, MATROSKA_ID_TAGTARGETS_CHAPTERUID))

            continue;



        ret = mkv_write_tag(s, ch->metadata, MATROSKA_ID_TAGTARGETS_CHAPTERUID, ch->id + mkv->chapter_id_offset, &mkv->tags);

        if (ret < 0) return ret;

    }



    if (mkv->have_attachments) {

        for (i = 0; i < mkv->attachments->num_entries; i++) {

            mkv_attachment *attachment = &mkv->attachments->entries[i];

            AVStream *st = s->streams[attachment->stream_idx];



            if (!mkv_check_tag(st->metadata, MATROSKA_ID_TAGTARGETS_ATTACHUID))

                continue;



            ret = mkv_write_tag(s, st->metadata, MATROSKA_ID_TAGTARGETS_ATTACHUID, attachment->fileuid, &mkv->tags);

            if (ret < 0)

                return ret;

        }

    }



    if (mkv->tags.pos) {

        if (s->pb->seekable && !mkv->is_live)

            put_ebml_void(s->pb, avio_tell(mkv->tags_bc));

        else

            end_ebml_master_crc32(s->pb, &mkv->tags_bc, mkv, mkv->tags);

    }

    return 0;

}
