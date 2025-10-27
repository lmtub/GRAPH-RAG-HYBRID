static int mkv_write_packet_internal(AVFormatContext *s, AVPacket *pkt, int add_cue)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVIOContext *pb         = s->pb;

    AVCodecParameters *par  = s->streams[pkt->stream_index]->codecpar;

    int keyframe            = !!(pkt->flags & AV_PKT_FLAG_KEY);

    int duration            = pkt->duration;

    int ret;

    int64_t ts = mkv->tracks[pkt->stream_index].write_dts ? pkt->dts : pkt->pts;

    int64_t relative_packet_pos;

    int dash_tracknum = mkv->is_dash ? mkv->dash_track_number : pkt->stream_index + 1;



    if (ts == AV_NOPTS_VALUE) {

        av_log(s, AV_LOG_ERROR, "Can't write packet with unknown timestamp\n");

        return AVERROR(EINVAL);

    }

    ts += mkv->tracks[pkt->stream_index].ts_offset;



    if (mkv->cluster_pos != -1) {

        int64_t cluster_time = ts - mkv->cluster_pts + mkv->tracks[pkt->stream_index].ts_offset;

        if ((int16_t)cluster_time != cluster_time) {

            av_log(s, AV_LOG_WARNING, "Starting new cluster due to timestamp\n");

            mkv_start_new_cluster(s, pkt);

        }

    }



    if (mkv->cluster_pos == -1) {

        mkv->cluster_pos = avio_tell(s->pb);

        ret = start_ebml_master_crc32(s->pb, &mkv->dyn_bc, &mkv->cluster, MATROSKA_ID_CLUSTER, 0);

        if (ret < 0)

            return ret;

        put_ebml_uint(mkv->dyn_bc, MATROSKA_ID_CLUSTERTIMECODE, FFMAX(0, ts));

        mkv->cluster_pts = FFMAX(0, ts);

    }

    pb = mkv->dyn_bc;



    relative_packet_pos = avio_tell(s->pb) - mkv->cluster.pos + avio_tell(pb);



    if (par->codec_type != AVMEDIA_TYPE_SUBTITLE) {

        mkv_write_block(s, pb, MATROSKA_ID_SIMPLEBLOCK, pkt, keyframe);

        if (s->pb->seekable && (par->codec_type == AVMEDIA_TYPE_VIDEO && keyframe || add_cue)) {

            ret = mkv_add_cuepoint(mkv->cues, pkt->stream_index, dash_tracknum, ts, mkv->cluster_pos, relative_packet_pos, -1);

            if (ret < 0) return ret;

        }

    } else {

        if (par->codec_id == AV_CODEC_ID_WEBVTT) {

            duration = mkv_write_vtt_blocks(s, pb, pkt);

        } else {

            ebml_master blockgroup = start_ebml_master(pb, MATROSKA_ID_BLOCKGROUP,

                                                       mkv_blockgroup_size(pkt->size));



#if FF_API_CONVERGENCE_DURATION

FF_DISABLE_DEPRECATION_WARNINGS

            /* For backward compatibility, prefer convergence_duration. */

            if (pkt->convergence_duration > 0) {

                duration = pkt->convergence_duration;

            }

FF_ENABLE_DEPRECATION_WARNINGS

#endif

            /* All subtitle blocks are considered to be keyframes. */

            mkv_write_block(s, pb, MATROSKA_ID_BLOCK, pkt, 1);

            put_ebml_uint(pb, MATROSKA_ID_BLOCKDURATION, duration);

            end_ebml_master(pb, blockgroup);

        }



        if (s->pb->seekable) {

            ret = mkv_add_cuepoint(mkv->cues, pkt->stream_index, dash_tracknum, ts,

                                   mkv->cluster_pos, relative_packet_pos, duration);

            if (ret < 0)

                return ret;

        }

    }



    mkv->duration = FFMAX(mkv->duration, ts + duration);



    if (mkv->stream_durations)

        mkv->stream_durations[pkt->stream_index] =

            FFMAX(mkv->stream_durations[pkt->stream_index], ts + duration);



    return 0;

}
