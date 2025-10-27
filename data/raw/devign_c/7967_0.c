static int mxf_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    MXFContext *mxf = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st = s->streams[pkt->stream_index];

    MXFStreamContext *sc = st->priv_data;

    MXFIndexEntry ie = {0};



    if (!mxf->edit_unit_byte_count && !(mxf->edit_units_count % EDIT_UNITS_PER_BODY)) {

        mxf->index_entries = av_realloc(mxf->index_entries,

            (mxf->edit_units_count + EDIT_UNITS_PER_BODY)*sizeof(*mxf->index_entries));

        if (!mxf->index_entries) {

            av_log(s, AV_LOG_ERROR, "could not allocate index entries\n");

            return -1;

        }

    }



    if (st->codec->codec_id == AV_CODEC_ID_MPEG2VIDEO) {

        if (!mxf_parse_mpeg2_frame(s, st, pkt, &ie)) {

            av_log(s, AV_LOG_ERROR, "could not get mpeg2 profile and level\n");

            return -1;

        }

    }



    if (!mxf->header_written) {

        if (mxf->edit_unit_byte_count) {

            mxf_write_partition(s, 1, 2, header_open_partition_key, 1);

            mxf_write_klv_fill(s);

            mxf_write_index_table_segment(s);

        } else {

            mxf_write_partition(s, 0, 0, header_open_partition_key, 1);

        }

        mxf->header_written = 1;

    }



    if (st->index == 0) {

        if (!mxf->edit_unit_byte_count &&

            (!mxf->edit_units_count || mxf->edit_units_count > EDIT_UNITS_PER_BODY) &&

            !(ie.flags & 0x33)) { // I frame, Gop start

            mxf_write_klv_fill(s);

            mxf_write_partition(s, 1, 2, body_partition_key, 0);



            mxf_write_klv_fill(s);

            mxf_write_index_table_segment(s);

        }



        mxf_write_klv_fill(s);

        mxf_write_system_item(s);



        if (!mxf->edit_unit_byte_count) {

            mxf->index_entries[mxf->edit_units_count].offset = mxf->body_offset;

            mxf->index_entries[mxf->edit_units_count].flags = ie.flags;

            mxf->index_entries[mxf->edit_units_count].temporal_ref = ie.temporal_ref;

            mxf->body_offset += KAG_SIZE; // size of system element

        }

        mxf->edit_units_count++;

    } else if (!mxf->edit_unit_byte_count && st->index == 1) {

        mxf->index_entries[mxf->edit_units_count-1].slice_offset =

            mxf->body_offset - mxf->index_entries[mxf->edit_units_count-1].offset;

    }



    mxf_write_klv_fill(s);

    avio_write(pb, sc->track_essence_element_key, 16); // write key

    if (s->oformat == &ff_mxf_d10_muxer) {

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

            mxf_write_d10_video_packet(s, st, pkt);

        else

            mxf_write_d10_audio_packet(s, st, pkt);

    } else {

        klv_encode_ber4_length(pb, pkt->size); // write length

        avio_write(pb, pkt->data, pkt->size);

        mxf->body_offset += 16+4+pkt->size + klv_fill_size(16+4+pkt->size);

    }



    avio_flush(pb);



    return 0;

}
