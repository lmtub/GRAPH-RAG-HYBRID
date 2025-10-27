static int mov_write_wave_tag(AVIOContext *pb, MOVTrack *track)

{

    int64_t pos = avio_tell(pb);



    avio_wb32(pb, 0);     /* size */

    ffio_wfourcc(pb, "wave");



    if (track->enc->codec_id != AV_CODEC_ID_QDM2) {

    avio_wb32(pb, 12);    /* size */

    ffio_wfourcc(pb, "frma");

    avio_wl32(pb, track->tag);

    }



    if (track->enc->codec_id == AV_CODEC_ID_AAC) {

        /* useless atom needed by mplayer, ipod, not needed by quicktime */

        avio_wb32(pb, 12); /* size */

        ffio_wfourcc(pb, "mp4a");

        avio_wb32(pb, 0);

        mov_write_esds_tag(pb, track);

    } else if (mov_pcm_le_gt16(track->enc->codec_id)) {

        mov_write_enda_tag(pb);

    } else if (track->enc->codec_id == AV_CODEC_ID_AMR_NB) {

        mov_write_amr_tag(pb, track);

    } else if (track->enc->codec_id == AV_CODEC_ID_AC3) {

        mov_write_ac3_tag(pb, track);

    } else if (track->enc->codec_id == AV_CODEC_ID_ALAC ||

               track->enc->codec_id == AV_CODEC_ID_QDM2) {

        mov_write_extradata_tag(pb, track);

    } else if (track->enc->codec_id == AV_CODEC_ID_ADPCM_MS ||

               track->enc->codec_id == AV_CODEC_ID_ADPCM_IMA_WAV) {

        mov_write_ms_tag(pb, track);

    }



    avio_wb32(pb, 8);     /* size */

    avio_wb32(pb, 0);     /* null tag */



    return update_size(pb, pos);

}
