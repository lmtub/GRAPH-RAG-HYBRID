static int avi_write_idx1(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    AVIContext *avi = s->priv_data;

    int64_t idx_chunk;

    int i;

    char tag[5];



    if (pb->seekable) {

        AVIStream *avist;

        AVIIentry *ie = 0, *tie;

        int empty, stream_id = -1;



        idx_chunk = ff_start_tag(pb, "idx1");

        for (i = 0; i < s->nb_streams; i++) {

            avist        = s->streams[i]->priv_data;

            avist->entry = 0;

        }



        do {

            empty = 1;

            for (i = 0; i < s->nb_streams; i++) {

                avist = s->streams[i]->priv_data;

                if (avist->indexes.entry <= avist->entry)

                    continue;



                tie = avi_get_ientry(&avist->indexes, avist->entry);

                if (empty || tie->pos < ie->pos) {

                    ie        = tie;

                    stream_id = i;

                }

                empty = 0;

            }

            if (!empty) {

                avist = s->streams[stream_id]->priv_data;

                avi_stream2fourcc(tag, stream_id,

                                  s->streams[stream_id]->codecpar->codec_type);

                ffio_wfourcc(pb, tag);

                avio_wl32(pb, ie->flags);

                avio_wl32(pb, ie->pos);

                avio_wl32(pb, ie->len);

                avist->entry++;

            }

        } while (!empty);

        ff_end_tag(pb, idx_chunk);



        avi_write_counters(s, avi->riff_id);

    }

    return 0;

}
