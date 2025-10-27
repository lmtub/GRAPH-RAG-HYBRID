static int gxf_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    GXFContext *gxf = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st = s->streams[pkt->stream_index];

    int64_t pos = avio_tell(pb);

    int padding = 0;

    int packet_start_offset = avio_tell(pb) / 1024;



    gxf_write_packet_header(pb, PKT_MEDIA);

    if (st->codec->codec_id == AV_CODEC_ID_MPEG2VIDEO && pkt->size % 4) /* MPEG-2 frames must be padded */

        padding = 4 - pkt->size % 4;

    else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO)

        padding = GXF_AUDIO_PACKET_SIZE - pkt->size;

    gxf_write_media_preamble(s, pkt, pkt->size + padding);

    avio_write(pb, pkt->data, pkt->size);

    gxf_write_padding(pb, padding);



    if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

        if (!(gxf->flt_entries_nb % 500)) {

            int err;

            if ((err = av_reallocp_array(&gxf->flt_entries,

                                         gxf->flt_entries_nb + 500,

                                         sizeof(*gxf->flt_entries))) < 0) {

                gxf->flt_entries_nb = 0;

                av_log(s, AV_LOG_ERROR, "could not reallocate flt entries\n");

                return err;

            }

        }

        gxf->flt_entries[gxf->flt_entries_nb++] = packet_start_offset;

        gxf->nb_fields += 2; // count fields

    }



    updatePacketSize(pb, pos);



    gxf->packet_count++;

    if (gxf->packet_count == 100) {

        gxf_write_map_packet(s, 0);

        gxf->packet_count = 0;

    }



    return 0;

}
