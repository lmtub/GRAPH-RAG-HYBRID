static int avi_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    unsigned char tag[5];

    unsigned int flags = 0;

    const int stream_index = pkt->stream_index;

    int size               = pkt->size;

    AVIContext *avi     = s->priv_data;

    AVIOContext *pb     = s->pb;

    AVIStream *avist    = s->streams[stream_index]->priv_data;

    AVCodecParameters *par = s->streams[stream_index]->codecpar;



    while (par->block_align == 0 && pkt->dts != AV_NOPTS_VALUE &&

           pkt->dts > avist->packet_count) {

        AVPacket empty_packet;



        av_init_packet(&empty_packet);

        empty_packet.size         = 0;

        empty_packet.data         = NULL;

        empty_packet.stream_index = stream_index;

        avi_write_packet(s, &empty_packet);

    }

    avist->packet_count++;



    // Make sure to put an OpenDML chunk when the file size exceeds the limits

    if (pb->seekable &&

        (avio_tell(pb) - avi->riff_start > AVI_MAX_RIFF_SIZE)) {

        avi_write_ix(s);

        ff_end_tag(pb, avi->movi_list);



        if (avi->riff_id == 1)

            avi_write_idx1(s);



        ff_end_tag(pb, avi->riff_start);

        avi->movi_list = avi_start_new_riff(s, pb, "AVIX", "movi");

    }



    avi_stream2fourcc(tag, stream_index, par->codec_type);

    if (pkt->flags & AV_PKT_FLAG_KEY)

        flags = 0x10;

    if (par->codec_type == AVMEDIA_TYPE_AUDIO)

        avist->audio_strm_length += size;



    if (s->pb->seekable) {

        int err;

        AVIIndex *idx = &avist->indexes;

        int cl = idx->entry / AVI_INDEX_CLUSTER_SIZE;

        int id = idx->entry % AVI_INDEX_CLUSTER_SIZE;

        if (idx->ents_allocated <= idx->entry) {

            if ((err = av_reallocp(&idx->cluster,

                                   (cl + 1) * sizeof(*idx->cluster))) < 0) {

                idx->ents_allocated = 0;

                idx->entry          = 0;

                return err;

            }

            idx->cluster[cl] =

                av_malloc(AVI_INDEX_CLUSTER_SIZE * sizeof(AVIIentry));

            if (!idx->cluster[cl])

                return -1;

            idx->ents_allocated += AVI_INDEX_CLUSTER_SIZE;

        }



        idx->cluster[cl][id].flags = flags;

        idx->cluster[cl][id].pos   = avio_tell(pb) - avi->movi_list;

        idx->cluster[cl][id].len   = size;

        idx->entry++;

    }



    avio_write(pb, tag, 4);

    avio_wl32(pb, size);

    avio_write(pb, pkt->data, size);

    if (size & 1)

        avio_w8(pb, 0);



    return 0;

}
