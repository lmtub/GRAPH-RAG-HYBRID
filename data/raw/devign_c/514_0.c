static int xmv_read_header(AVFormatContext *s)

{

    XMVDemuxContext *xmv = s->priv_data;

    AVIOContext     *pb  = s->pb;



    uint32_t file_version;

    uint32_t this_packet_size;

    uint16_t audio_track;

    int ret;



    s->ctx_flags |= AVFMTCTX_NOHEADER;



    avio_skip(pb, 4); /* Next packet size */



    this_packet_size = avio_rl32(pb);



    avio_skip(pb, 4); /* Max packet size */

    avio_skip(pb, 4); /* "xobX" */



    file_version = avio_rl32(pb);

    if ((file_version != 4) && (file_version != 2))

        avpriv_request_sample(s, "Uncommon version %"PRIu32"", file_version);



    /* Video tracks */



    xmv->video_width    = avio_rl32(pb);

    xmv->video_height   = avio_rl32(pb);

    xmv->video_duration = avio_rl32(pb);



    /* Audio tracks */



    xmv->audio_track_count = avio_rl16(pb);



    avio_skip(pb, 2); /* Unknown (padding?) */



    xmv->audio = av_mallocz_array(xmv->audio_track_count, sizeof(XMVAudioPacket));

    if (!xmv->audio) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    for (audio_track = 0; audio_track < xmv->audio_track_count; audio_track++) {

        XMVAudioPacket *packet = &xmv->audio[audio_track];



        packet->compression     = avio_rl16(pb);

        packet->channels        = avio_rl16(pb);

        packet->sample_rate     = avio_rl32(pb);

        packet->bits_per_sample = avio_rl16(pb);

        packet->flags           = avio_rl16(pb);



        packet->bit_rate      = packet->bits_per_sample *

                                packet->sample_rate *

                                packet->channels;

        packet->block_align   = XMV_BLOCK_ALIGN_SIZE * packet->channels;

        packet->block_samples = 64;

        packet->codec_id      = ff_wav_codec_get_id(packet->compression,

                                                    packet->bits_per_sample);



        packet->stream_index = -1;



        packet->frame_size  = 0;

        packet->block_count = 0;



        /* TODO: ADPCM'd 5.1 sound is encoded in three separate streams.

         *       Those need to be interleaved to a proper 5.1 stream. */

        if (packet->flags & XMV_AUDIO_ADPCM51)

            av_log(s, AV_LOG_WARNING, "Unsupported 5.1 ADPCM audio stream "

                                      "(0x%04X)\n", packet->flags);



        if (!packet->channels || !packet->sample_rate ||

             packet->channels >= UINT16_MAX / XMV_BLOCK_ALIGN_SIZE) {

            av_log(s, AV_LOG_ERROR, "Invalid parameters for audio track %"PRIu16".\n",

                   audio_track);

            ret = AVERROR_INVALIDDATA;

            goto fail;

        }

    }





    /* Initialize the packet context */



    xmv->next_packet_offset = avio_tell(pb);

    xmv->next_packet_size   = this_packet_size - xmv->next_packet_offset;

    xmv->stream_count       = xmv->audio_track_count + 1;



    return 0;



fail:

    xmv_read_close(s);

    return ret;

}
