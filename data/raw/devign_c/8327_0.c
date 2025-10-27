static int64_t get_dts(AVFormatContext *s, int64_t pos)

{

    AVIOContext *pb = s->pb;

    int64_t dts;



    ffm_seek1(s, pos);

    avio_skip(pb, 4);

    dts = avio_rb64(pb);

    av_dlog(s, "dts=%0.6f\n", dts / 1000000.0);

    return dts;

}
