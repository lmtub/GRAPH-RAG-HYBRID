static int r3d_read_redv(AVFormatContext *s, AVPacket *pkt, Atom *atom)

{

    AVStream *st = s->streams[0];

    int tmp;

    int av_unused tmp2;

    uint64_t pos = avio_tell(s->pb);

    unsigned dts;

    int ret;



    dts = avio_rb32(s->pb);



    tmp = avio_rb32(s->pb);

    av_dlog(s, "frame num %d\n", tmp);



    tmp  = avio_r8(s->pb); // major version

    tmp2 = avio_r8(s->pb); // minor version

    av_dlog(s, "version %d.%d\n", tmp, tmp2);



    tmp = avio_rb16(s->pb); // unknown

    av_dlog(s, "unknown %d\n", tmp);



    if (tmp > 4) {

        tmp = avio_rb16(s->pb); // unknown

        av_dlog(s, "unknown %d\n", tmp);



        tmp = avio_rb16(s->pb); // unknown

        av_dlog(s, "unknown %d\n", tmp);



        tmp = avio_rb32(s->pb);

        av_dlog(s, "width %d\n", tmp);

        tmp = avio_rb32(s->pb);

        av_dlog(s, "height %d\n", tmp);



        tmp = avio_rb32(s->pb);

        av_dlog(s, "metadata len %d\n", tmp);

    }

    tmp = atom->size - 8 - (avio_tell(s->pb) - pos);

    if (tmp < 0)

        return -1;

    ret = av_get_packet(s->pb, pkt, tmp);

    if (ret < 0) {

        av_log(s, AV_LOG_ERROR, "error reading video packet\n");

        return -1;

    }



    pkt->stream_index = 0;

    pkt->dts = dts;

    if (st->avg_frame_rate.num)

        pkt->duration = (uint64_t)st->time_base.den*

            st->avg_frame_rate.den/st->avg_frame_rate.num;

    av_dlog(s, "pkt dts %"PRId64" duration %d\n", pkt->dts, pkt->duration);



    return 0;

}
