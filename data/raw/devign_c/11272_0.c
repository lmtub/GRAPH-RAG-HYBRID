static int load_ipmovie_packet(IPMVEContext *s, AVIOContext *pb,

    AVPacket *pkt) {



    int chunk_type;



    if (s->audio_chunk_offset && s->audio_channels && s->audio_bits) {

        if (s->audio_type == AV_CODEC_ID_NONE) {

            av_log(s->avf, AV_LOG_ERROR, "Can not read audio packet before"

                   "audio codec is known\n");

                return CHUNK_BAD;

        }



        /* adjust for PCM audio by skipping chunk header */

        if (s->audio_type != AV_CODEC_ID_INTERPLAY_DPCM) {

            s->audio_chunk_offset += 6;

            s->audio_chunk_size -= 6;

        }



        avio_seek(pb, s->audio_chunk_offset, SEEK_SET);

        s->audio_chunk_offset = 0;



        if (s->audio_chunk_size != av_get_packet(pb, pkt, s->audio_chunk_size))

            return CHUNK_EOF;



        pkt->stream_index = s->audio_stream_index;

        pkt->pts = s->audio_frame_count;



        /* audio frame maintenance */

        if (s->audio_type != AV_CODEC_ID_INTERPLAY_DPCM)

            s->audio_frame_count +=

            (s->audio_chunk_size / s->audio_channels / (s->audio_bits / 8));

        else

            s->audio_frame_count +=

                (s->audio_chunk_size - 6 - s->audio_channels) / s->audio_channels;



        av_log(s->avf, AV_LOG_TRACE, "sending audio frame with pts %"PRId64" (%d audio frames)\n",

                pkt->pts, s->audio_frame_count);



        chunk_type = CHUNK_VIDEO;



    } else if (s->decode_map_chunk_offset) {



        /* send both the decode map and the video data together */



        if (av_new_packet(pkt, 2 + s->decode_map_chunk_size + s->video_chunk_size))

            return CHUNK_NOMEM;



        if (s->has_palette) {

            uint8_t *pal;



            pal = av_packet_new_side_data(pkt, AV_PKT_DATA_PALETTE,

                                          AVPALETTE_SIZE);

            if (pal) {

                memcpy(pal, s->palette, AVPALETTE_SIZE);

                s->has_palette = 0;

            }

        }



        if (s->changed) {

            ff_add_param_change(pkt, 0, 0, 0, s->video_width, s->video_height);

            s->changed = 0;

        }

        pkt->pos= s->decode_map_chunk_offset;

        avio_seek(pb, s->decode_map_chunk_offset, SEEK_SET);

        s->decode_map_chunk_offset = 0;



        AV_WL16(pkt->data, s->decode_map_chunk_size);

        if (avio_read(pb, pkt->data + 2, s->decode_map_chunk_size) !=

            s->decode_map_chunk_size) {

            av_packet_unref(pkt);

            return CHUNK_EOF;

        }



        avio_seek(pb, s->video_chunk_offset, SEEK_SET);

        s->video_chunk_offset = 0;



        if (avio_read(pb, pkt->data + 2 + s->decode_map_chunk_size,

            s->video_chunk_size) != s->video_chunk_size) {

            av_packet_unref(pkt);

            return CHUNK_EOF;

        }



        pkt->stream_index = s->video_stream_index;

        pkt->pts = s->video_pts;



        av_log(s->avf, AV_LOG_TRACE, "sending video frame with pts %"PRId64"\n", pkt->pts);



        s->video_pts += s->frame_pts_inc;



        chunk_type = CHUNK_VIDEO;



    } else {



        avio_seek(pb, s->next_chunk_offset, SEEK_SET);

        chunk_type = CHUNK_DONE;



    }



    return chunk_type;

}
