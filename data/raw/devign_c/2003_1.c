static void mxf_write_generic_sound_common(AVFormatContext *s, AVStream *st, const UID key, unsigned size)

{

    AVIOContext *pb = s->pb;



    mxf_write_generic_desc(s, st, key, size+5+12+8+8);



    // audio locked

    mxf_write_local_tag(pb, 1, 0x3D02);

    avio_w8(pb, 1);



    // write audio sampling rate

    mxf_write_local_tag(pb, 8, 0x3D03);

    avio_wb32(pb, st->codec->sample_rate);

    avio_wb32(pb, 1);



    mxf_write_local_tag(pb, 4, 0x3D07);

    avio_wb32(pb, st->codec->channels);



    mxf_write_local_tag(pb, 4, 0x3D01);

    avio_wb32(pb, av_get_bits_per_sample(st->codec->codec_id));

}
