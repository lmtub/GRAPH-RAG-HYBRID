static int xmv_process_packet_header(AVFormatContext *s)

{

    XMVDemuxContext *xmv = s->priv_data;

    AVIOContext     *pb  = s->pb;



    uint8_t  data[8];

    uint16_t audio_track;

    uint64_t data_offset;



    /* Next packet size */

    xmv->next_packet_size = avio_rl32(pb);



    /* Packet video header */



    if (avio_read(pb, data, 8) != 8)

        return AVERROR(EIO);



    xmv->video.data_size     = AV_RL32(data) & 0x007FFFFF;



    xmv->video.current_frame = 0;

    xmv->video.frame_count   = (AV_RL32(data) >> 23) & 0xFF;



    xmv->video.has_extradata = (data[3] & 0x80) != 0;



    /* Adding the audio data sizes and the video data size keeps you 4 bytes

     * short for every audio track. But as playing around with XMV files with

     * ADPCM audio showed, taking the extra 4 bytes from the audio data gives

     * you either completely distorted audio or click (when skipping the

     * remaining 68 bytes of the ADPCM block). Subtracting 4 bytes for every

     * audio track from the video data works at least for the audio. Probably

     * some alignment thing?

     * The video data has (always?) lots of padding, so it should work out...

     */

    xmv->video.data_size -= xmv->audio_track_count * 4;



    xmv->current_stream = 0;

    if (!xmv->video.frame_count) {

        xmv->video.frame_count = 1;

        xmv->current_stream    = xmv->stream_count > 1;

    }



    /* Packet audio header */



    for (audio_track = 0; audio_track < xmv->audio_track_count; audio_track++) {

        XMVAudioPacket *packet = &xmv->audio[audio_track];



        if (avio_read(pb, data, 4) != 4)

            return AVERROR(EIO);



        packet->data_size = AV_RL32(data) & 0x007FFFFF;

        if ((packet->data_size == 0) && (audio_track != 0))

            /* This happens when I create an XMV with several identical audio

             * streams. From the size calculations, duplicating the previous

             * stream's size works out, but the track data itself is silent.

             * Maybe this should also redirect the offset to the previous track?

             */

            packet->data_size = xmv->audio[audio_track - 1].data_size;



        /* Carve up the audio data in frame_count slices */

        packet->frame_size  = packet->data_size  / xmv->video.frame_count;

        packet->frame_size -= packet->frame_size % packet->block_align;

    }



    /* Packet data offsets */



    data_offset = avio_tell(pb);



    xmv->video.data_offset = data_offset;

    data_offset += xmv->video.data_size;



    for (audio_track = 0; audio_track < xmv->audio_track_count; audio_track++) {

        xmv->audio[audio_track].data_offset = data_offset;

        data_offset += xmv->audio[audio_track].data_size;

    }



    /* Video frames header */



    /* Read new video extra data */

    if (xmv->video.data_size > 0) {

        if (xmv->video.has_extradata) {

            xmv_read_extradata(xmv->video.extradata, pb);



            xmv->video.data_size   -= 4;

            xmv->video.data_offset += 4;



            if (xmv->video.stream_index >= 0) {

                AVStream *vst = s->streams[xmv->video.stream_index];



                av_assert0(xmv->video.stream_index < s->nb_streams);



                if (vst->codec->extradata_size < 4) {

                    av_freep(&vst->codec->extradata);



                    ff_alloc_extradata(vst->codec, 4);

                }



                memcpy(vst->codec->extradata, xmv->video.extradata, 4);

            }

        }

    }



    return 0;

}
