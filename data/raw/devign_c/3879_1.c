static int matroska_parse_block(MatroskaDemuxContext *matroska, uint8_t *data,

                                int size, int64_t pos, uint64_t cluster_time,

                                uint64_t block_duration, int is_keyframe,

                                int64_t cluster_pos)

{

    uint64_t timecode = AV_NOPTS_VALUE;

    MatroskaTrack *track;

    int res = 0;

    AVStream *st;

    int16_t block_time;

    uint32_t *lace_size = NULL;

    int n, flags, laces = 0;

    uint64_t num, duration;



    if ((n = matroska_ebmlnum_uint(matroska, data, size, &num)) < 0) {

        av_log(matroska->ctx, AV_LOG_ERROR, "EBML block data error\n");

        return n;

    }

    data += n;

    size -= n;



    track = matroska_find_track_by_num(matroska, num);

    if (!track || !track->stream) {

        av_log(matroska->ctx, AV_LOG_INFO,

               "Invalid stream %"PRIu64" or size %u\n", num, size);

        return AVERROR_INVALIDDATA;

    } else if (size <= 3)

        return 0;

    st = track->stream;

    if (st->discard >= AVDISCARD_ALL)

        return res;



    block_time = AV_RB16(data);

    data += 2;

    flags = *data++;

    size -= 3;

    if (is_keyframe == -1)

        is_keyframe = flags & 0x80 ? AV_PKT_FLAG_KEY : 0;



    if (cluster_time != (uint64_t)-1

        && (block_time >= 0 || cluster_time >= -block_time)) {

        timecode = cluster_time + block_time;

        if (track->type == MATROSKA_TRACK_TYPE_SUBTITLE

            && timecode < track->end_timecode)

            is_keyframe = 0;  /* overlapping subtitles are not key frame */

        if (is_keyframe)

            av_add_index_entry(st, cluster_pos, timecode, 0,0,AVINDEX_KEYFRAME);

    }



    if (matroska->skip_to_keyframe && track->type != MATROSKA_TRACK_TYPE_SUBTITLE) {

        if (!is_keyframe || timecode < matroska->skip_to_timecode)

            return res;

        matroska->skip_to_keyframe = 0;

    }



    res = matroska_parse_laces(matroska, &data, &size, (flags & 0x06) >> 1,

                               &lace_size, &laces);



    if (res)

        goto end;



    if (block_duration != AV_NOPTS_VALUE) {

        duration = block_duration / laces;

        if (block_duration != duration * laces) {

            av_log(matroska->ctx, AV_LOG_WARNING,

                   "Incorrect block_duration, possibly corrupted container");

        }

    } else {

        duration = track->default_duration / matroska->time_scale;

        block_duration = duration * laces;

    }



    if (timecode != AV_NOPTS_VALUE)

        track->end_timecode =

            FFMAX(track->end_timecode, timecode + block_duration);



    for (n = 0; n < laces; n++) {

        if ((st->codec->codec_id == AV_CODEC_ID_RA_288 ||

             st->codec->codec_id == AV_CODEC_ID_COOK ||

             st->codec->codec_id == AV_CODEC_ID_SIPR ||

             st->codec->codec_id == AV_CODEC_ID_ATRAC3) &&

             st->codec->block_align && track->audio.sub_packet_size) {



            res = matroska_parse_rm_audio(matroska, track, st, data, size,

                                          timecode, duration, pos);

            if (res)

                goto end;



        } else {

            res = matroska_parse_frame(matroska, track, st, data, lace_size[n],

                                      timecode, duration,

                                      pos, !n? is_keyframe : 0);

            if (res)

                goto end;

        }



        if (timecode != AV_NOPTS_VALUE)

            timecode = duration ? timecode + duration : AV_NOPTS_VALUE;

        data += lace_size[n];

        size -= lace_size[n];

    }



end:

    av_free(lace_size);

    return res;

}
