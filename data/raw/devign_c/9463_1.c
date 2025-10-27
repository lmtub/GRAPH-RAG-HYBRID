static int matroska_parse_block(MatroskaDemuxContext *matroska, uint8_t *data,

                                int size, int64_t pos, uint64_t cluster_time,

                                uint64_t block_duration, int is_keyframe,

                                uint8_t *additional, uint64_t additional_id, int additional_size,

                                int64_t cluster_pos)

{

    uint64_t timecode = AV_NOPTS_VALUE;

    MatroskaTrack *track;

    int res = 0;

    AVStream *st;

    int16_t block_time;

    uint32_t *lace_size = NULL;

    int n, flags, laces = 0;

    uint64_t num;



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

    av_assert1(block_duration != AV_NOPTS_VALUE);



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

        if (timecode < matroska->skip_to_timecode)

            return res;

        if (!st->skip_to_keyframe) {

            av_log(matroska->ctx, AV_LOG_ERROR, "File is broken, keyframes not correctly marked!\n");

            matroska->skip_to_keyframe = 0;

        }

        if (is_keyframe)

            matroska->skip_to_keyframe = 0;

    }



    res = matroska_parse_laces(matroska, &data, size, (flags & 0x06) >> 1,

                               &lace_size, &laces);



    if (res)

        goto end;



    if (!block_duration)

        block_duration = track->default_duration * laces / matroska->time_scale;



    if (cluster_time != (uint64_t)-1 && (block_time >= 0 || cluster_time >= -block_time))

        track->end_timecode =

            FFMAX(track->end_timecode, timecode + block_duration);



    for (n = 0; n < laces; n++) {

        int64_t lace_duration = block_duration*(n+1) / laces - block_duration*n / laces;



        if (lace_size[n] > size) {

            av_log(matroska->ctx, AV_LOG_ERROR, "Invalid packet size\n");

            break;

        }



        if ((st->codec->codec_id == AV_CODEC_ID_RA_288 ||

             st->codec->codec_id == AV_CODEC_ID_COOK ||

             st->codec->codec_id == AV_CODEC_ID_SIPR ||

             st->codec->codec_id == AV_CODEC_ID_ATRAC3) &&

             st->codec->block_align && track->audio.sub_packet_size) {



            res = matroska_parse_rm_audio(matroska, track, st, data, size,

                                          timecode, pos);

            if (res)

                goto end;



        } else {

            res = matroska_parse_frame(matroska, track, st, data, lace_size[n],

                                      timecode, lace_duration,

                                      pos, !n? is_keyframe : 0,

                                      additional, additional_id, additional_size);

            if (res)

                goto end;

        }



        if (timecode != AV_NOPTS_VALUE)

            timecode = lace_duration ? timecode + lace_duration : AV_NOPTS_VALUE;

        data += lace_size[n];

        size -= lace_size[n];

    }



end:

    av_free(lace_size);

    return res;

}
