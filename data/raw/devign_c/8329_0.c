static int write_packet(AVFormatContext *s, AVPacket *pkt)

{

    WVMuxContext *wc = s->priv_data;

    AVCodecContext *codec = s->streams[0]->codec;

    AVIOContext *pb = s->pb;

    uint64_t size;

    uint32_t flags;

    uint32_t left = pkt->size;

    uint8_t *ptr = pkt->data;

    int off = codec->channels > 2 ? 4 : 0;



    /* FIXME: Simplify decoder/demuxer so bellow code can support midstream

     *        change of stream parameters */

    wc->duration += pkt->duration;

    ffio_wfourcc(pb, "wvpk");

    if (off) {

        size = AV_RL32(pkt->data);

        if (size <= 12)

            return AVERROR_INVALIDDATA;

        size -= 12;

    } else {

        size = pkt->size;

    }



    if (size + off > left)

        return AVERROR_INVALIDDATA;



    avio_wl32(pb, size + 12);

    avio_wl16(pb, 0x410);

    avio_w8(pb, 0);

    avio_w8(pb, 0);

    avio_wl32(pb, -1);

    avio_wl32(pb, pkt->pts);

    ptr += off; left -= off;

    flags = AV_RL32(ptr + 4);

    avio_write(pb, ptr, size);

    ptr += size; left -= size;



    while (!(flags & WV_END_BLOCK) &&

            (left >= 4 + WV_EXTRA_SIZE)) {

        ffio_wfourcc(pb, "wvpk");

        size = AV_RL32(ptr);

        ptr += 4; left -= 4;

        if (size < 24 || size - 24 > left)

            return AVERROR_INVALIDDATA;

        avio_wl32(pb, size);

        avio_wl16(pb, 0x410);

        avio_w8(pb, 0);

        avio_w8(pb, 0);

        avio_wl32(pb, -1);

        avio_wl32(pb, pkt->pts);

        flags = AV_RL32(ptr + 4);

        avio_write(pb, ptr, WV_EXTRA_SIZE);

        ptr += WV_EXTRA_SIZE; left -= WV_EXTRA_SIZE;

        avio_write(pb, ptr, size - 24);

        ptr += size - 24; left -= size - 24;

    }



    return 0;

}
