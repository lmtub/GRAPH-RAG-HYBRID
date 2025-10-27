static void write_odml_master(AVFormatContext *s, int stream_index)

{

    AVIOContext *pb = s->pb;

    AVStream *st = s->streams[stream_index];

    AVCodecContext *enc = st->codec;

    AVIStream *avist = st->priv_data;

    unsigned char tag[5];

    int j;



    /* Starting to lay out AVI OpenDML master index.

        * We want to make it JUNK entry for now, since we'd

        * like to get away without making AVI an OpenDML one

        * for compatibility reasons. */

    avist->indexes.entry      = avist->indexes.ents_allocated = 0;

    avist->indexes.indx_start = ff_start_tag(pb, "JUNK");

    avio_wl16(pb, 4);   /* wLongsPerEntry */

    avio_w8(pb, 0);     /* bIndexSubType (0 == frame index) */

    avio_w8(pb, 0);     /* bIndexType (0 == AVI_INDEX_OF_INDEXES) */

    avio_wl32(pb, 0);   /* nEntriesInUse (will fill out later on) */

    ffio_wfourcc(pb, avi_stream2fourcc(tag, stream_index, enc->codec_type));

                        /* dwChunkId */

    avio_wl64(pb, 0);   /* dwReserved[3] */

    avio_wl32(pb, 0);   /* Must be 0.    */

    for (j = 0; j < AVI_MASTER_INDEX_SIZE * 2; j++)

        avio_wl64(pb, 0);

    ff_end_tag(pb, avist->indexes.indx_start);

}
