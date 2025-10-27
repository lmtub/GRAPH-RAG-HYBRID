static int mkv_write_packet_internal(AVFormatContext *s, AVPacket *pkt)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVIOContext *pb = s->pb;

    AVCodecContext *codec = s->streams[pkt->stream_index]->codec;

    int keyframe = !!(pkt->flags & AV_PKT_FLAG_KEY);

    int duration = pkt->duration;

    int ret;

    int64_t ts = mkv->tracks[pkt->stream_index].write_dts ? pkt->dts : pkt->pts;



    if (ts == AV_NOPTS_VALUE) {

        av_log(s, AV_LOG_ERROR, "Can't write packet with unknown timestamp\n");

        return AVERROR(EINVAL);

    }



    if (!s->pb->seekable) {

        if (!mkv->dyn_bc)

            avio_open_dyn_buf(&mkv->dyn_bc);

        pb = mkv->dyn_bc;

    }



    if (mkv->cluster_pos == -1) {

        mkv->cluster_pos = avio_tell(s->pb);

        mkv->cluster = start_ebml_master(pb, MATROSKA_ID_CLUSTER, 0);

        put_ebml_uint(pb, MATROSKA_ID_CLUSTERTIMECODE, FFMAX(0, ts));

        mkv->cluster_pts = FFMAX(0, ts);

    }



    if (codec->codec_type != AVMEDIA_TYPE_SUBTITLE) {

        mkv_write_block(s, pb, MATROSKA_ID_SIMPLEBLOCK, pkt, keyframe << 7);

    } else if (codec->codec_id == AV_CODEC_ID_SSA) {

        duration = mkv_write_ass_blocks(s, pb, pkt);

    } else if (codec->codec_id == AV_CODEC_ID_SRT) {

        duration = mkv_write_srt_blocks(s, pb, pkt);

    } else {

        ebml_master blockgroup = start_ebml_master(pb, MATROSKA_ID_BLOCKGROUP, mkv_blockgroup_size(pkt->size));

        /* For backward compatibility, prefer convergence_duration. */

        if (pkt->convergence_duration > 0) {

            duration = pkt->convergence_duration;

        }

        mkv_write_block(s, pb, MATROSKA_ID_BLOCK, pkt, 0);

        put_ebml_uint(pb, MATROSKA_ID_BLOCKDURATION, duration);

        end_ebml_master(pb, blockgroup);

    }



    if (codec->codec_type == AVMEDIA_TYPE_VIDEO && keyframe) {

        ret = mkv_add_cuepoint(mkv->cues, pkt->stream_index, ts, mkv->cluster_pos);

        if (ret < 0) return ret;

    }



    mkv->duration = FFMAX(mkv->duration, ts + duration);

    return 0;

}
