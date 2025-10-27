static void mov_read_chapters(AVFormatContext *s)

{

    MOVContext *mov = s->priv_data;

    AVStream *st = NULL;

    MOVStreamContext *sc;

    int64_t cur_pos;

    int i;



    for (i = 0; i < s->nb_streams; i++)

        if (s->streams[i]->id == mov->chapter_track) {

            st = s->streams[i];

            break;

        }

    if (!st) {

        av_log(s, AV_LOG_ERROR, "Referenced QT chapter track not found\n");

        return;

    }



    st->discard = AVDISCARD_ALL;

    sc = st->priv_data;

    cur_pos = avio_tell(sc->pb);



    for (i = 0; i < st->nb_index_entries; i++) {

        AVIndexEntry *sample = &st->index_entries[i];

        int64_t end = i+1 < st->nb_index_entries ? st->index_entries[i+1].timestamp : st->duration;

        uint8_t *title;

        uint16_t ch;

        int len, title_len;



        if (avio_seek(sc->pb, sample->pos, SEEK_SET) != sample->pos) {

            av_log(s, AV_LOG_ERROR, "Chapter %d not found in file\n", i);

            goto finish;

        }



        // the first two bytes are the length of the title

        len = avio_rb16(sc->pb);

        if (len > sample->size-2)

            continue;

        title_len = 2*len + 1;

        if (!(title = av_mallocz(title_len)))

            goto finish;



        // The samples could theoretically be in any encoding if there's an encd

        // atom following, but in practice are only utf-8 or utf-16, distinguished

        // instead by the presence of a BOM

        ch = avio_rb16(sc->pb);

        if (ch == 0xfeff)

            avio_get_str16be(sc->pb, len, title, title_len);

        else if (ch == 0xfffe)

            avio_get_str16le(sc->pb, len, title, title_len);

        else {

            AV_WB16(title, ch);

            avio_get_str(sc->pb, len - 2, title + 2, title_len - 2);

        }



        ff_new_chapter(s, i, st->time_base, sample->timestamp, end, title);

        av_freep(&title);

    }

finish:

    avio_seek(sc->pb, cur_pos, SEEK_SET);

}
