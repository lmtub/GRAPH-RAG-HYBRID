static int rtmp_write(URLContext *s, const uint8_t *buf, int size)
{
    RTMPContext *rt = s->priv_data;
    int size_temp = size;
    int pktsize, pkttype;
    uint32_t ts;
    const uint8_t *buf_temp = buf;
    uint8_t c;
    int ret;
    do {
        if (rt->skip_bytes) {
            int skip = FFMIN(rt->skip_bytes, size_temp);
            buf_temp       += skip;
            size_temp      -= skip;
            rt->skip_bytes -= skip;
            continue;
        if (rt->flv_header_bytes < 11) {
            const uint8_t *header = rt->flv_header;
            int copy = FFMIN(11 - rt->flv_header_bytes, size_temp);
            bytestream_get_buffer(&buf_temp, rt->flv_header + rt->flv_header_bytes, copy);
            rt->flv_header_bytes += copy;
            size_temp            -= copy;
            if (rt->flv_header_bytes < 11)
                break;
            pkttype = bytestream_get_byte(&header);
            pktsize = bytestream_get_be24(&header);
            ts = bytestream_get_be24(&header);
            ts |= bytestream_get_byte(&header) << 24;
            bytestream_get_be24(&header);
            rt->flv_size = pktsize;
            //force 12bytes header
            if (((pkttype == RTMP_PT_VIDEO || pkttype == RTMP_PT_AUDIO) && ts == 0) ||
                pkttype == RTMP_PT_NOTIFY) {
                if (pkttype == RTMP_PT_NOTIFY)
                    pktsize += 16;
                rt->prev_pkt[1][RTMP_SOURCE_CHANNEL].channel_id = 0;
            //this can be a big packet, it's better to send it right here
            if ((ret = ff_rtmp_packet_create(&rt->out_pkt, RTMP_SOURCE_CHANNEL,
                                             pkttype, ts, pktsize)) < 0)
            rt->out_pkt.extra = rt->main_channel_id;
            rt->flv_data = rt->out_pkt.data;
            if (pkttype == RTMP_PT_NOTIFY)
                ff_amf_write_string(&rt->flv_data, "@setDataFrame");
        if (rt->flv_size - rt->flv_off > size_temp) {
            bytestream_get_buffer(&buf_temp, rt->flv_data + rt->flv_off, size_temp);
            rt->flv_off += size_temp;
            size_temp = 0;
        } else {
            bytestream_get_buffer(&buf_temp, rt->flv_data + rt->flv_off, rt->flv_size - rt->flv_off);
            size_temp   -= rt->flv_size - rt->flv_off;
            rt->flv_off += rt->flv_size - rt->flv_off;
        if (rt->flv_off == rt->flv_size) {
            rt->skip_bytes = 4;
            if ((ret = ff_rtmp_packet_write(rt->stream, &rt->out_pkt,
                                            rt->chunk_size, rt->prev_pkt[1])) < 0)
            ff_rtmp_packet_destroy(&rt->out_pkt);
            rt->flv_size = 0;
            rt->flv_off = 0;
            rt->flv_header_bytes = 0;
    } while (buf_temp - buf < size);