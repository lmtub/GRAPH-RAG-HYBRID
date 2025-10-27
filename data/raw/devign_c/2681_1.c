int ff_mov_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    MOVMuxContext *mov = s->priv_data;

    AVIOContext *pb = s->pb;

    MOVTrack *trk = &mov->tracks[pkt->stream_index];

    AVCodecContext *enc = trk->enc;

    unsigned int samples_in_chunk = 0;

    int size = pkt->size;

    uint8_t *reformatted_data = NULL;



    if (mov->flags & FF_MOV_FLAG_FRAGMENT) {

        int ret;

        if (mov->fragments > 0) {

            if (!trk->mdat_buf) {

                if ((ret = avio_open_dyn_buf(&trk->mdat_buf)) < 0)

                    return ret;

            }

            pb = trk->mdat_buf;

        } else {

            if (!mov->mdat_buf) {

                if ((ret = avio_open_dyn_buf(&mov->mdat_buf)) < 0)

                    return ret;

            }

            pb = mov->mdat_buf;

        }

    }



    if (enc->codec_id == AV_CODEC_ID_AMR_NB) {

        /* We must find out how many AMR blocks there are in one packet */

        static uint16_t packed_size[16] =

            {13, 14, 16, 18, 20, 21, 27, 32, 6, 0, 0, 0, 0, 0, 0, 1};

        int len = 0;



        while (len < size && samples_in_chunk < 100) {

            len += packed_size[(pkt->data[len] >> 3) & 0x0F];

            samples_in_chunk++;

        }

        if (samples_in_chunk > 1) {

            av_log(s, AV_LOG_ERROR, "fatal error, input is not a single packet, implement a AVParser for it\n");

            return -1;

        }

    } else if (trk->sample_size)

        samples_in_chunk = size / trk->sample_size;

    else

        samples_in_chunk = 1;



    /* copy extradata if it exists */

    if (trk->vos_len == 0 && enc->extradata_size > 0) {

        trk->vos_len  = enc->extradata_size;

        trk->vos_data = av_malloc(trk->vos_len);

        memcpy(trk->vos_data, enc->extradata, trk->vos_len);

    }



    if (enc->codec_id == AV_CODEC_ID_H264 && trk->vos_len > 0 && *(uint8_t *)trk->vos_data != 1) {

        /* from x264 or from bytestream h264 */

        /* nal reformating needed */

        if (trk->hint_track >= 0 && trk->hint_track < mov->nb_streams) {

            ff_avc_parse_nal_units_buf(pkt->data, &reformatted_data,

                                       &size);

            avio_write(pb, reformatted_data, size);

        } else {

            size = ff_avc_parse_nal_units(pb, pkt->data, pkt->size);

        }

    } else if (enc->codec_id == AV_CODEC_ID_HEVC && trk->vos_len > 6 &&

               (AV_RB24(trk->vos_data) == 1 || AV_RB32(trk->vos_data) == 1)) {

        /* extradata is Annex B, assume the bitstream is too and convert it */

        if (trk->hint_track >= 0 && trk->hint_track < mov->nb_streams) {

            ff_hevc_annexb2mp4_buf(pkt->data, &reformatted_data, &size, 0, NULL);

            avio_write(pb, reformatted_data, size);

        } else {

            size = ff_hevc_annexb2mp4(pb, pkt->data, pkt->size, 0, NULL);

        }

    } else {

        avio_write(pb, pkt->data, size);

    }



    if ((enc->codec_id == AV_CODEC_ID_DNXHD ||

         enc->codec_id == AV_CODEC_ID_AC3) && !trk->vos_len) {

        /* copy frame to create needed atoms */

        trk->vos_len  = size;

        trk->vos_data = av_malloc(size);

        if (!trk->vos_data)

            return AVERROR(ENOMEM);

        memcpy(trk->vos_data, pkt->data, size);

    }



    if (trk->entry >= trk->cluster_capacity) {

        unsigned new_capacity = 2 * (trk->entry + MOV_INDEX_CLUSTER_SIZE);

        if (av_reallocp_array(&trk->cluster, new_capacity,

                              sizeof(*trk->cluster)))

            return AVERROR(ENOMEM);

        trk->cluster_capacity = new_capacity;

    }



    trk->cluster[trk->entry].pos              = avio_tell(pb) - size;

    trk->cluster[trk->entry].samples_in_chunk = samples_in_chunk;

    trk->cluster[trk->entry].size             = size;

    trk->cluster[trk->entry].entries          = samples_in_chunk;

    trk->cluster[trk->entry].dts              = pkt->dts;

    if (!trk->entry && trk->start_dts != AV_NOPTS_VALUE) {

        /* First packet of a new fragment. We already wrote the duration

         * of the last packet of the previous fragment based on track_duration,

         * which might not exactly match our dts. Therefore adjust the dts

         * of this packet to be what the previous packets duration implies. */

        trk->cluster[trk->entry].dts = trk->start_dts + trk->track_duration;

    }

    if (!trk->entry && trk->start_dts == AV_NOPTS_VALUE && !mov->use_editlist &&

        s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_MAKE_ZERO) {

        /* Not using edit lists and shifting the first track to start from zero.

         * If the other streams start from a later timestamp, we won't be able

         * to signal the difference in starting time without an edit list.

         * Thus move the timestamp for this first sample to 0, increasing

         * its duration instead. */

        trk->cluster[trk->entry].dts = trk->start_dts = 0;

    }

    if (trk->start_dts == AV_NOPTS_VALUE) {

        trk->start_dts = pkt->dts;

        if (pkt->dts && mov->flags & FF_MOV_FLAG_EMPTY_MOOV)

            av_log(s, AV_LOG_WARNING,

                   "Track %d starts with a nonzero dts %"PRId64". This "

                   "currently isn't handled correctly in combination with "

                   "empty_moov.\n", pkt->stream_index, pkt->dts);

    }

    trk->track_duration = pkt->dts - trk->start_dts + pkt->duration;



    if (pkt->pts == AV_NOPTS_VALUE) {

        av_log(s, AV_LOG_WARNING, "pts has no value\n");

        pkt->pts = pkt->dts;

    }

    if (pkt->dts != pkt->pts)

        trk->flags |= MOV_TRACK_CTTS;

    trk->cluster[trk->entry].cts   = pkt->pts - pkt->dts;

    trk->cluster[trk->entry].flags = 0;

    if (enc->codec_id == AV_CODEC_ID_VC1) {

        mov_parse_vc1_frame(pkt, trk, mov->fragments);

    } else if (pkt->flags & AV_PKT_FLAG_KEY) {

        if (mov->mode == MODE_MOV && enc->codec_id == AV_CODEC_ID_MPEG2VIDEO &&

            trk->entry > 0) { // force sync sample for the first key frame

            mov_parse_mpeg2_frame(pkt, &trk->cluster[trk->entry].flags);

            if (trk->cluster[trk->entry].flags & MOV_PARTIAL_SYNC_SAMPLE)

                trk->flags |= MOV_TRACK_STPS;

        } else {

            trk->cluster[trk->entry].flags = MOV_SYNC_SAMPLE;

        }

        if (trk->cluster[trk->entry].flags & MOV_SYNC_SAMPLE)

            trk->has_keyframes++;

    }

    trk->entry++;

    trk->sample_count += samples_in_chunk;

    mov->mdat_size    += size;



    if (trk->hint_track >= 0 && trk->hint_track < mov->nb_streams)

        ff_mov_add_hinted_packet(s, pkt, trk->hint_track, trk->entry,

                                 reformatted_data, size);

    av_free(reformatted_data);

    return 0;

}
