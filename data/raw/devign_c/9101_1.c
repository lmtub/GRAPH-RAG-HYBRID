static int mov_write_stbl_tag(AVIOContext *pb, MOVTrack *track)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "stbl");

    mov_write_stsd_tag(pb, track);

    mov_write_stts_tag(pb, track);

    if ((track->enc->codec_type == AVMEDIA_TYPE_VIDEO ||

         track->enc->codec_tag == MKTAG('r','t','p',' ')) &&

        track->has_keyframes && track->has_keyframes < track->entry)

        mov_write_stss_tag(pb, track, MOV_SYNC_SAMPLE);

    if (track->mode == MODE_MOV && track->flags & MOV_TRACK_STPS)

        mov_write_stss_tag(pb, track, MOV_PARTIAL_SYNC_SAMPLE);

    if (track->enc->codec_type == AVMEDIA_TYPE_VIDEO &&

        track->flags & MOV_TRACK_CTTS && track->entry)

        mov_write_ctts_tag(pb, track);

    mov_write_stsc_tag(pb, track);

    mov_write_stsz_tag(pb, track);

    mov_write_stco_tag(pb, track);

    return update_size(pb, pos);

}
