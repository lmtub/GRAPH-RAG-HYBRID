static void gxf_read_index(AVFormatContext *s, int pkt_len) {

    AVIOContext *pb = s->pb;

    AVStream *st = s->streams[0];

    uint32_t fields_per_map = avio_rl32(pb);

    uint32_t map_cnt = avio_rl32(pb);

    int i;

    pkt_len -= 8;

    if (s->flags & AVFMT_FLAG_IGNIDX) {

        avio_skip(pb, pkt_len);

        return;

    }

    if (map_cnt > 1000) {

        av_log(s, AV_LOG_ERROR, "too many index entries %u (%x)\n", map_cnt, map_cnt);

        map_cnt = 1000;

    }

    if (pkt_len < 4 * map_cnt) {

        av_log(s, AV_LOG_ERROR, "invalid index length\n");

        avio_skip(pb, pkt_len);

        return;

    }

    pkt_len -= 4 * map_cnt;

    av_add_index_entry(st, 0, 0, 0, 0, 0);

    for (i = 0; i < map_cnt; i++)

        av_add_index_entry(st, (uint64_t)avio_rl32(pb) * 1024,

                           i * (uint64_t)fields_per_map + 1, 0, 0, 0);

    avio_skip(pb, pkt_len);

}
