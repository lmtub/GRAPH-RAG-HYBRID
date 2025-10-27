static int mov_write_stbl_tag(ByteIOContext *pb, MOVTrack* track)

{

    offset_t pos = url_ftell(pb);

    put_be32(pb, 0); /* size */

    put_tag(pb, "stbl");

    mov_write_stsd_tag(pb, track);

    mov_write_stts_tag(pb, track);

    if (track->enc->codec_type == CODEC_TYPE_VIDEO &&

        track->hasKeyframes < track->entry)

        mov_write_stss_tag(pb, track);

    if (track->enc->codec_type == CODEC_TYPE_VIDEO &&

        track->hasBframes)

        mov_write_ctts_tag(pb, track);

    mov_write_stsc_tag(pb, track);

    mov_write_stsz_tag(pb, track);

    mov_write_stco_tag(pb, track);

    return updateSize(pb, pos);

}
