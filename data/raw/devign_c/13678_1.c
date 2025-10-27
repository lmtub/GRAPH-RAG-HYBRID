static int mov_write_trak_tag(AVIOContext *pb, MOVMuxContext *mov,

                              MOVTrack *track, AVStream *st)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "trak");

    mov_write_tkhd_tag(pb, mov, track, st);

    if (track->mode == MODE_PSP || track->flags & MOV_TRACK_CTTS ||

        (track->entry && track->cluster[0].dts) ||

        is_clcp_track(track)) {

        if (mov->use_editlist)

            mov_write_edts_tag(pb, mov, track);  // PSP Movies require edts box

        else if ((track->entry && track->cluster[0].dts) || track->mode == MODE_PSP || is_clcp_track(track))

            av_log(mov->fc, AV_LOG_WARNING,

                   "Not writing any edit list even though one would have been required\n");

    }

    if (track->tref_tag)

        mov_write_tref_tag(pb, track);

    mov_write_mdia_tag(pb, mov, track);

    if (track->mode == MODE_PSP)

        mov_write_uuid_tag_psp(pb, track); // PSP Movies require this uuid box

    if (track->tag == MKTAG('r','t','p',' '))

        mov_write_udta_sdp(pb, track);

    if (track->mode == MODE_MOV) {

        if (track->enc->codec_type == AVMEDIA_TYPE_VIDEO) {

            double sample_aspect_ratio = av_q2d(st->sample_aspect_ratio);

            if ((0.0 != sample_aspect_ratio && 1.0 != sample_aspect_ratio)) {

                mov_write_tapt_tag(pb, track);

            }

        }

        if (is_clcp_track(track)) {

            mov_write_tapt_tag(pb, track);

        }

    }

    mov_write_track_udta_tag(pb, mov, st);

    return update_size(pb, pos);

}
