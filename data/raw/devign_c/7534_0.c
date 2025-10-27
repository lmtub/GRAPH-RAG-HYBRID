static int mov_write_audio_tag(ByteIOContext *pb, MOVTrack* track)

{

    int pos = url_ftell(pb);

    put_be32(pb, 0); /* size */



    if(track->enc->codec_id == CODEC_ID_PCM_MULAW)

      put_tag(pb, "ulaw");

    else if(track->enc->codec_id == CODEC_ID_PCM_ALAW)

      put_tag(pb, "alaw");

    else if(track->enc->codec_id == CODEC_ID_ADPCM_IMA_QT)

      put_tag(pb, "ima4");

    else if(track->enc->codec_id == CODEC_ID_MACE3)

      put_tag(pb, "MAC3");

    else if(track->enc->codec_id == CODEC_ID_MACE6)

      put_tag(pb, "MAC6");

    else if(track->enc->codec_id == CODEC_ID_AAC)

      put_tag(pb, "mp4a");

    else if(track->enc->codec_id == CODEC_ID_AMR_NB)

      put_tag(pb, "samr");

    else

      put_tag(pb, "    ");



    put_be32(pb, 0); /* Reserved */

    put_be16(pb, 0); /* Reserved */

    put_be16(pb, 1); /* Data-reference index, XXX  == 1 */

    /* SoundDescription */

    put_be16(pb, 0); /* Version */

    put_be16(pb, 0); /* Revision level */

    put_be32(pb, 0); /* Reserved */



    put_be16(pb, track->enc->channels); /* Number of channels */

    /* TODO: Currently hard-coded to 16-bit, there doesn't seem

                 to be a good way to get number of bits of audio */

    put_be16(pb, 0x10); /* Reserved */

    put_be16(pb, 0); /* compression ID (= 0) */

    put_be16(pb, 0); /* packet size (= 0) */

    put_be16(pb, track->timescale); /* Time scale */

    put_be16(pb, 0); /* Reserved */



    if(track->enc->codec_id == CODEC_ID_AAC)

        mov_write_esds_tag(pb, track);

    if(track->enc->codec_id == CODEC_ID_AMR_NB)

        mov_write_damr_tag(pb);

    return updateSize (pb, pos);

}
