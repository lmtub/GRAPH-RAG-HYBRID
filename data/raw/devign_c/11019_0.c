static int mov_write_ms_tag(AVIOContext *pb, MOVTrack *track)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0);

    avio_wl32(pb, track->tag); // store it byteswapped

    track->enc->codec_tag = av_bswap16(track->tag >> 16);

    ff_put_wav_header(pb, track->enc, 0);

    return update_size(pb, pos);

}
