int ff_ape_write_tag(AVFormatContext *s)

{

    AVDictionaryEntry *e = NULL;

    int64_t start, end;

    int size, count = 0;



    if (!s->pb->seekable)

        return 0;



    start = avio_tell(s->pb);



    // header

    avio_write(s->pb, "APETAGEX", 8);   // id

    avio_wl32 (s->pb, APE_TAG_VERSION); // version

    avio_wl32(s->pb, 0);                // reserve space for size

    avio_wl32(s->pb, 0);                // reserve space for tag count



    // flags

    avio_wl32(s->pb, APE_TAG_FLAG_CONTAINS_HEADER | APE_TAG_FLAG_CONTAINS_FOOTER |

                     APE_TAG_FLAG_IS_HEADER);

    ffio_fill(s->pb, 0, 8);             // reserved



    while ((e = av_dict_get(s->metadata, "", e, AV_DICT_IGNORE_SUFFIX))) {

        int val_len = strlen(e->value);



        avio_wl32(s->pb, val_len);            // value length

        avio_wl32(s->pb, 0);                  // item flags

        avio_put_str(s->pb, e->key);          // key

        avio_write(s->pb, e->value, val_len); // value

        count++;

    }



    size = avio_tell(s->pb) - start;



    // footer

    avio_write(s->pb, "APETAGEX", 8);   // id

    avio_wl32 (s->pb, APE_TAG_VERSION); // version

    avio_wl32(s->pb, size);             // size

    avio_wl32(s->pb, count);            // tag count



    // flags

    avio_wl32(s->pb, APE_TAG_FLAG_CONTAINS_HEADER | APE_TAG_FLAG_CONTAINS_FOOTER);

    ffio_fill(s->pb, 0, 8);             // reserved



    // update values in the header

    end = avio_tell(s->pb);

    avio_seek(s->pb, start + 12, SEEK_SET);

    avio_wl32(s->pb, size);

    avio_wl32(s->pb, count);

    avio_seek(s->pb, end, SEEK_SET);



    return 0;

}
