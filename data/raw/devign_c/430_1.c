static int mpegts_read_header(AVFormatContext *s)

{

    MpegTSContext *ts = s->priv_data;

    AVIOContext *pb   = s->pb;

    uint8_t buf[8 * 1024] = {0};

    int len;

    int64_t pos, probesize = s->probesize;



    if (ffio_ensure_seekback(pb, probesize) < 0)

        av_log(s, AV_LOG_WARNING, "Failed to allocate buffers for seekback\n");



    /* read the first 8192 bytes to get packet size */

    pos = avio_tell(pb);

    len = avio_read(pb, buf, sizeof(buf));

    ts->raw_packet_size = get_packet_size(buf, len);

    if (ts->raw_packet_size <= 0) {

        av_log(s, AV_LOG_WARNING, "Could not detect TS packet size, defaulting to non-FEC/DVHS\n");

        ts->raw_packet_size = TS_PACKET_SIZE;

    }

    ts->stream     = s;

    ts->auto_guess = 0;



    if (s->iformat == &ff_mpegts_demuxer) {

        /* normal demux */



        /* first do a scan to get all the services */

        seek_back(s, pb, pos);



        mpegts_open_section_filter(ts, SDT_PID, sdt_cb, ts, 1);



        mpegts_open_section_filter(ts, PAT_PID, pat_cb, ts, 1);



        handle_packets(ts, probesize / ts->raw_packet_size);

        /* if could not find service, enable auto_guess */



        ts->auto_guess = 1;



        av_log(ts->stream, AV_LOG_TRACE, "tuning done\n");



        s->ctx_flags |= AVFMTCTX_NOHEADER;

    } else {

        AVStream *st;

        int pcr_pid, pid, nb_packets, nb_pcrs, ret, pcr_l;

        int64_t pcrs[2], pcr_h;

        int packet_count[2];

        uint8_t packet[TS_PACKET_SIZE];

        const uint8_t *data;



        /* only read packets */



        st = avformat_new_stream(s, NULL);

        if (!st)

            return AVERROR(ENOMEM);

        avpriv_set_pts_info(st, 60, 1, 27000000);

        st->codecpar->codec_type = AVMEDIA_TYPE_DATA;

        st->codecpar->codec_id   = AV_CODEC_ID_MPEG2TS;



        /* we iterate until we find two PCRs to estimate the bitrate */

        pcr_pid    = -1;

        nb_pcrs    = 0;

        nb_packets = 0;

        for (;;) {

            ret = read_packet(s, packet, ts->raw_packet_size, &data);

            if (ret < 0)

                return ret;

            pid = AV_RB16(data + 1) & 0x1fff;

            if ((pcr_pid == -1 || pcr_pid == pid) &&

                parse_pcr(&pcr_h, &pcr_l, data) == 0) {

                finished_reading_packet(s, ts->raw_packet_size);

                pcr_pid = pid;

                packet_count[nb_pcrs] = nb_packets;

                pcrs[nb_pcrs] = pcr_h * 300 + pcr_l;

                nb_pcrs++;

                if (nb_pcrs >= 2)

                    break;

            } else {

                finished_reading_packet(s, ts->raw_packet_size);

            }

            nb_packets++;

        }



        /* NOTE1: the bitrate is computed without the FEC */

        /* NOTE2: it is only the bitrate of the start of the stream */

        ts->pcr_incr = (pcrs[1] - pcrs[0]) / (packet_count[1] - packet_count[0]);

        ts->cur_pcr  = pcrs[0] - ts->pcr_incr * packet_count[0];

        s->bit_rate  = TS_PACKET_SIZE * 8 * 27000000LL / ts->pcr_incr;

        st->codecpar->bit_rate = s->bit_rate;

        st->start_time      = ts->cur_pcr;

        av_log(ts->stream, AV_LOG_TRACE, "start=%0.3f pcr=%0.3f incr=%d\n",

                st->start_time / 1000000.0, pcrs[0] / 27e6, ts->pcr_incr);

    }



    seek_back(s, pb, pos);

    return 0;

}
