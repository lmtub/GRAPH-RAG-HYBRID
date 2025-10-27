static int r3d_read_reda(AVFormatContext *s, AVPacket *pkt, Atom *atom)

{

    AVStream *st = s->streams[1];

    int av_unused tmp, tmp2;

    int samples, size;

    uint64_t pos = avio_tell(s->pb);

    unsigned dts;

    int ret;



    dts = avio_rb32(s->pb);



    st->codec->sample_rate = avio_rb32(s->pb);



    samples = avio_rb32(s->pb);



    tmp = avio_rb32(s->pb);

    av_dlog(s, "packet num %d\n", tmp);



    tmp = avio_rb16(s->pb); // unknown

    av_dlog(s, "unknown %d\n", tmp);



    tmp  = avio_r8(s->pb); // major version

    tmp2 = avio_r8(s->pb); // minor version

    av_dlog(s, "version %d.%d\n", tmp, tmp2);



    tmp = avio_rb32(s->pb); // unknown

    av_dlog(s, "unknown %d\n", tmp);



    size = atom->size - 8 - (avio_tell(s->pb) - pos);

    if (size < 0)

        return -1;

    ret = av_get_packet(s->pb, pkt, size);

    if (ret < 0) {

        av_log(s, AV_LOG_ERROR, "error reading audio packet\n");

        return ret;

    }



    pkt->stream_index = 1;

    pkt->dts = dts;

    pkt->duration = av_rescale(samples, st->time_base.den, st->codec->sample_rate);

    av_dlog(s, "pkt dts %"PRId64" duration %d samples %d sample rate %d\n",

            pkt->dts, pkt->duration, samples, st->codec->sample_rate);



    return 0;

}
