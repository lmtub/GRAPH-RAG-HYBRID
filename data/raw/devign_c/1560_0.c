static void mkv_write_block(AVFormatContext *s, AVIOContext *pb,

                            unsigned int blockid, AVPacket *pkt, int flags)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVCodecContext *codec = s->streams[pkt->stream_index]->codec;

    uint8_t *data = NULL, *side_data = NULL;

    int offset = 0, size = pkt->size, side_data_size = 0;

    int64_t ts = mkv->tracks[pkt->stream_index].write_dts ? pkt->dts : pkt->pts;

    uint64_t additional_id = 0;

    int64_t discard_padding = 0;

    ebml_master block_group, block_additions, block_more;



    av_log(s, AV_LOG_DEBUG, "Writing block at offset %" PRIu64 ", size %d, "

           "pts %" PRId64 ", dts %" PRId64 ", duration %d, flags %d\n",

           avio_tell(pb), pkt->size, pkt->pts, pkt->dts, pkt->duration, flags);

    if (codec->codec_id == AV_CODEC_ID_H264 && codec->extradata_size > 0 &&

        (AV_RB24(codec->extradata) == 1 || AV_RB32(codec->extradata) == 1))

        ff_avc_parse_nal_units_buf(pkt->data, &data, &size);

    else if (codec->codec_id == AV_CODEC_ID_HEVC && codec->extradata_size > 6 &&

             (AV_RB24(codec->extradata) == 1 || AV_RB32(codec->extradata) == 1))

        /* extradata is Annex B, assume the bitstream is too and convert it */

        ff_hevc_annexb2mp4_buf(pkt->data, &data, &size, 0, NULL);

    else if (codec->codec_id == AV_CODEC_ID_WAVPACK) {

        int ret = mkv_strip_wavpack(pkt->data, &data, &size);

        if (ret < 0) {

            av_log(s, AV_LOG_ERROR, "Error stripping a WavPack packet.\n");

            return;

        }

    } else

        data = pkt->data;



    if (codec->codec_id == AV_CODEC_ID_PRORES) {

        /* Matroska specification requires to remove the first QuickTime atom

         */

        size  -= 8;

        offset = 8;

    }



    side_data = av_packet_get_side_data(pkt,

                                        AV_PKT_DATA_SKIP_SAMPLES,

                                        &side_data_size);



    if (side_data && side_data_size >= 10) {

        discard_padding = av_rescale_q(AV_RL32(side_data + 4),

                                       (AVRational){1, codec->sample_rate},

                                       (AVRational){1, 1000000000});

    }



    side_data = av_packet_get_side_data(pkt,

                                        AV_PKT_DATA_MATROSKA_BLOCKADDITIONAL,

                                        &side_data_size);

    if (side_data) {

        additional_id = AV_RB64(side_data);

        side_data += 8;

        side_data_size -= 8;

    }



    if ((side_data_size && additional_id == 1) || discard_padding) {

        block_group = start_ebml_master(pb, MATROSKA_ID_BLOCKGROUP, 0);

        blockid = MATROSKA_ID_BLOCK;

    }



    put_ebml_id(pb, blockid);

    put_ebml_num(pb, size + 4, 0);

    // this assumes stream_index is less than 126

    avio_w8(pb, 0x80 | (mkv->is_dash ? mkv->dash_track_number : (pkt->stream_index + 1)));

    avio_wb16(pb, ts - mkv->cluster_pts);

    avio_w8(pb, flags);

    avio_write(pb, data + offset, size);

    if (data != pkt->data)

        av_free(data);



    if (discard_padding) {

        put_ebml_sint(pb, MATROSKA_ID_DISCARDPADDING, discard_padding);

    }



    if (side_data_size && additional_id == 1) {

        block_additions = start_ebml_master(pb, MATROSKA_ID_BLOCKADDITIONS, 0);

        block_more = start_ebml_master(pb, MATROSKA_ID_BLOCKMORE, 0);

        put_ebml_uint(pb, MATROSKA_ID_BLOCKADDID, 1);

        put_ebml_id(pb, MATROSKA_ID_BLOCKADDITIONAL);

        put_ebml_num(pb, side_data_size, 0);

        avio_write(pb, side_data, side_data_size);

        end_ebml_master(pb, block_more);

        end_ebml_master(pb, block_additions);

    }

    if ((side_data_size && additional_id == 1) || discard_padding) {

        end_ebml_master(pb, block_group);

    }

}
