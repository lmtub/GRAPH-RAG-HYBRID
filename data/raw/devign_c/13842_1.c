static int mkv_write_trailer(AVFormatContext *s)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVIOContext *pb = s->pb;

    int64_t currentpos, cuespos;

    int ret;



    // check if we have an audio packet cached

    if (mkv->cur_audio_pkt.size > 0) {

        ret = mkv_write_packet_internal(s, &mkv->cur_audio_pkt, 0);

        av_free_packet(&mkv->cur_audio_pkt);

        if (ret < 0) {

            av_log(s, AV_LOG_ERROR,

                   "Could not write cached audio packet ret:%d\n", ret);

            return ret;

        }

    }



    if (mkv->dyn_bc) {

        end_ebml_master(mkv->dyn_bc, mkv->cluster);

        mkv_flush_dynbuf(s);

    } else if (mkv->cluster_pos != -1) {

        end_ebml_master(pb, mkv->cluster);

    }



    if (mkv->mode != MODE_WEBM) {

        ret = mkv_write_chapters(s);

        if (ret < 0)

            return ret;

    }



    if (pb->seekable) {

        if (mkv->cues->num_entries) {

            if (mkv->reserve_cues_space) {

                int64_t cues_end;



                currentpos = avio_tell(pb);

                avio_seek(pb, mkv->cues_pos, SEEK_SET);



                cuespos  = mkv_write_cues(s, mkv->cues, mkv->tracks, s->nb_streams);

                cues_end = avio_tell(pb);

                if (cues_end > cuespos + mkv->reserve_cues_space) {

                    av_log(s, AV_LOG_ERROR,

                           "Insufficient space reserved for cues: %d "

                           "(needed: %" PRId64 ").\n",

                           mkv->reserve_cues_space, cues_end - cuespos);

                    return AVERROR(EINVAL);

                }



                if (cues_end < cuespos + mkv->reserve_cues_space)

                    put_ebml_void(pb, mkv->reserve_cues_space -

                                  (cues_end - cuespos));



                avio_seek(pb, currentpos, SEEK_SET);

            } else {

                cuespos = mkv_write_cues(s, mkv->cues, mkv->tracks, s->nb_streams);

            }



            ret = mkv_add_seekhead_entry(mkv->main_seekhead, MATROSKA_ID_CUES,

                                         cuespos);

            if (ret < 0)

                return ret;

        }



        mkv_write_seekhead(pb, mkv->main_seekhead);



        // update the duration

        av_log(s, AV_LOG_DEBUG, "end duration = %" PRIu64 "\n", mkv->duration);

        currentpos = avio_tell(pb);

        avio_seek(pb, mkv->duration_offset, SEEK_SET);

        put_ebml_float(pb, MATROSKA_ID_DURATION, mkv->duration);



        // update stream durations

        if (mkv->stream_durations) {

            int i;

            for (i = 0; i < s->nb_streams; ++i) {

                AVStream *st = s->streams[i];

                double duration_sec = mkv->stream_durations[i] * av_q2d(st->time_base);

                char duration_string[20] = "";



                av_log(s, AV_LOG_DEBUG, "stream %d end duration = %" PRIu64 "\n", i,

                       mkv->stream_durations[i]);



                if (!mkv->is_live && mkv->stream_duration_offsets[i] > 0) {

                    avio_seek(pb, mkv->stream_duration_offsets[i], SEEK_SET);



                    snprintf(duration_string, 20, "%02d:%02d:%012.9f",

                             (int) duration_sec / 3600, ((int) duration_sec / 60) % 60,

                             fmod(duration_sec, 60));



                    put_ebml_binary(pb, MATROSKA_ID_TAGSTRING, duration_string, 20);

                }

            }

        }



        avio_seek(pb, currentpos, SEEK_SET);

    }



    if (!mkv->is_live) {

        end_ebml_master(pb, mkv->segment);

    }

    av_freep(&mkv->tracks);

    av_freep(&mkv->cues->entries);

    av_freep(&mkv->cues);

    av_freep(&mkv->stream_durations);

    av_freep(&mkv->stream_duration_offsets);



    return 0;

}
