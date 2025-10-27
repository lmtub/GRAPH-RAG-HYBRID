static int hnm_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    Hnm4DemuxContext *hnm = s->priv_data;

    AVIOContext *pb = s->pb;

    int ret = 0;



    uint32_t superchunk_size, chunk_size;

    uint16_t chunk_id;



    if (hnm->currentframe == hnm->frames || pb->eof_reached)

        return AVERROR_EOF;



    if (hnm->superchunk_remaining == 0) {

        /* parse next superchunk */

        superchunk_size = avio_rl24(pb);

        avio_skip(pb, 1);



        hnm->superchunk_remaining = superchunk_size - 4;

    }



    chunk_size = avio_rl24(pb);

    avio_skip(pb, 1);

    chunk_id = avio_rl16(pb);

    avio_skip(pb, 2);



    if (chunk_size > hnm->superchunk_remaining) {

        av_log(s, AV_LOG_ERROR, "invalid chunk size: %u, offset: %u\n",

               chunk_size, (int) avio_tell(pb));

        avio_skip(pb, hnm->superchunk_remaining - 8);

        hnm->superchunk_remaining = 0;

    }



    switch (chunk_id) {

    case HNM4_CHUNK_ID_PL:

    case HNM4_CHUNK_ID_IZ:

    case HNM4_CHUNK_ID_IU:

        avio_seek(pb, -8, SEEK_CUR);

        ret += av_get_packet(pb, pkt, chunk_size);

        hnm->superchunk_remaining -= chunk_size;

        if (chunk_id == HNM4_CHUNK_ID_IZ || chunk_id == HNM4_CHUNK_ID_IU)

            hnm->currentframe++;

        break;



    case HNM4_CHUNK_ID_SD:

        avio_skip(pb, chunk_size - 8);

        hnm->superchunk_remaining -= chunk_size;

        break;



    default:

        av_log(s, AV_LOG_WARNING, "unknown chunk found: %d, offset: %d\n",

               chunk_id, (int) avio_tell(pb));

        avio_skip(pb, chunk_size - 8);

        hnm->superchunk_remaining -= chunk_size;

        break;

    }



    return ret;

}
