static int mov_write_stts_tag(AVIOContext *pb, MOVTrack *track)

{

    MOVStts *stts_entries;

    uint32_t entries = -1;

    uint32_t atom_size;

    int i;



    if (track->enc->codec_type == AVMEDIA_TYPE_AUDIO && !track->audio_vbr) {

        stts_entries = av_malloc(sizeof(*stts_entries)); /* one entry */

        if (!stts_entries)

            return AVERROR(ENOMEM);

        stts_entries[0].count = track->sample_count;

        stts_entries[0].duration = 1;

        entries = 1;

    } else {

        stts_entries = track->entry ?

                       av_malloc_array(track->entry, sizeof(*stts_entries)) : /* worst case */

                       NULL;

        if (!stts_entries)

            return AVERROR(ENOMEM);

        for (i = 0; i < track->entry; i++) {

            int duration = get_cluster_duration(track, i);

            if (i && duration == stts_entries[entries].duration) {

                stts_entries[entries].count++; /* compress */

            } else {

                entries++;

                stts_entries[entries].duration = duration;

                stts_entries[entries].count = 1;

            }

        }

        entries++; /* last one */

    }

    atom_size = 16 + (entries * 8);

    avio_wb32(pb, atom_size); /* size */

    ffio_wfourcc(pb, "stts");

    avio_wb32(pb, 0); /* version & flags */

    avio_wb32(pb, entries); /* entry count */

    for (i = 0; i < entries; i++) {

        avio_wb32(pb, stts_entries[i].count);

        avio_wb32(pb, stts_entries[i].duration);

    }

    av_free(stts_entries);

    return atom_size;

}
