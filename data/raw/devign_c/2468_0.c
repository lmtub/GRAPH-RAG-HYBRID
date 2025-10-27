static int mov_write_wfex_tag(AVIOContext *pb, MOVTrack *track)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0);

    ffio_wfourcc(pb, "wfex");

    ff_put_wav_header(pb, track->enc, FF_PUT_WAV_HEADER_FORCE_WAVEFORMATEX);

    return update_size(pb, pos);

}
