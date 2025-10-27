static int rpl_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    RPLContext *rpl = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream* stream;

    AVIndexEntry* index_entry;

    uint32_t ret;



    if (rpl->chunk_part == s->nb_streams) {

        rpl->chunk_number++;

        rpl->chunk_part = 0;

    }



    stream = s->streams[rpl->chunk_part];



    if (rpl->chunk_number >= stream->nb_index_entries)

        return AVERROR_EOF;



    index_entry = &stream->index_entries[rpl->chunk_number];



    if (rpl->frame_in_part == 0)

        if (avio_seek(pb, index_entry->pos, SEEK_SET) < 0)

            return AVERROR(EIO);



    if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

        stream->codec->codec_tag == 124) {

        // We have to split Escape 124 frames because there are

        // multiple frames per chunk in Escape 124 samples.

        uint32_t frame_size;



        avio_skip(pb, 4); /* flags */

        frame_size = avio_rl32(pb);

        if (avio_seek(pb, -8, SEEK_CUR) < 0)

            return AVERROR(EIO);



        ret = av_get_packet(pb, pkt, frame_size);

        if (ret != frame_size) {

            av_free_packet(pkt);

            return AVERROR(EIO);

        }

        pkt->duration = 1;

        pkt->pts = index_entry->timestamp + rpl->frame_in_part;

        pkt->stream_index = rpl->chunk_part;



        rpl->frame_in_part++;

        if (rpl->frame_in_part == rpl->frames_per_chunk) {

            rpl->frame_in_part = 0;

            rpl->chunk_part++;

        }

    } else {

        ret = av_get_packet(pb, pkt, index_entry->size);

        if (ret != index_entry->size) {

            av_free_packet(pkt);

            return AVERROR(EIO);

        }



        if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            // frames_per_chunk should always be one here; the header

            // parsing will warn if it isn't.

            pkt->duration = rpl->frames_per_chunk;

        } else {

            // All the audio codecs supported in this container

            // (at least so far) are constant-bitrate.

            pkt->duration = ret * 8;

        }

        pkt->pts = index_entry->timestamp;

        pkt->stream_index = rpl->chunk_part;

        rpl->chunk_part++;

    }



    // None of the Escape formats have keyframes, and the ADPCM

    // format used doesn't have keyframes.

    if (rpl->chunk_number == 0 && rpl->frame_in_part == 0)

        pkt->flags |= AV_PKT_FLAG_KEY;



    return ret;

}
