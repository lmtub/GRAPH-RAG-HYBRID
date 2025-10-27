static int qdm2_parse_packet(AVFormatContext *s, PayloadContext *qdm,

                             AVStream *st, AVPacket *pkt,

                             uint32_t *timestamp,

                             const uint8_t *buf, int len, int flags)

{

    int res = AVERROR_INVALIDDATA, n;

    const uint8_t *end = buf + len, *p = buf;



    if (len > 0) {

        if (len < 2)

            return AVERROR_INVALIDDATA;



        /* configuration block */

        if (*p == 0xff) {

            if (qdm->n_pkts > 0) {

                av_log(s, AV_LOG_WARNING,

                       "Out of sequence config - dropping queue\n");

                qdm->n_pkts = 0;

                memset(qdm->len, 0, sizeof(qdm->len));

            }



            if ((res = qdm2_parse_config(qdm, st, ++p, end)) < 0)

                return res;

            p += res;



            /* We set codec_id to CODEC_ID_NONE initially to

             * delay decoder initialization since extradata is

             * carried within the RTP stream, not SDP. Here,

             * by setting codec_id to CODEC_ID_QDM2, we are signalling

             * to the decoder that it is OK to initialize. */

            st->codec->codec_id = CODEC_ID_QDM2;

        }





        /* subpackets */

        while (end - p >= 4) {

            if ((res = qdm2_parse_subpacket(qdm, st, p, end)) < 0)

                return res;

            p += res;

        }



        qdm->timestamp = *timestamp;

        if (++qdm->n_pkts < qdm->subpkts_per_block)


        qdm->cache = 0;

        for (n = 0; n < 0x80; n++)

            if (qdm->len[n] > 0)

                qdm->cache++;

    }



    /* output the subpackets into freshly created superblock structures */

    if (!qdm->cache || (res = qdm2_restore_block(qdm, st, pkt)) < 0)

        return res;

    if (--qdm->cache == 0)

        qdm->n_pkts = 0;



    *timestamp = qdm->timestamp;

    qdm->timestamp = RTP_NOTS_VALUE;



    return (qdm->cache > 0) ? 1 : 0;

}