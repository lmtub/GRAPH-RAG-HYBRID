int ff_mpegts_parse_packet(MpegTSContext *ts, AVPacket *pkt,

                        const uint8_t *buf, int len)

{

    int len1;



    len1 = len;

    ts->pkt = pkt;

    ts->stop_parse = 0;

    for(;;) {

        if (ts->stop_parse>0)

            break;

        if (len < TS_PACKET_SIZE)

            return -1;

        if (buf[0] != 0x47) {

            buf++;

            len--;

        } else {

            handle_packet(ts, buf);

            buf += TS_PACKET_SIZE;

            len -= TS_PACKET_SIZE;

        }

    }

    return len1 - len;

}
