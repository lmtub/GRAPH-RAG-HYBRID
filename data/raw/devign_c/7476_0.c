matroska_parse_block(MatroskaDemuxContext *matroska, uint8_t *data, int size,

                     int64_t pos, uint64_t cluster_time,

                     int is_keyframe, int is_bframe,

                     int *ptrack, AVPacket **ppkt)

{

    int res = 0;

    int track;

    AVPacket *pkt;

    uint8_t *origdata = data;

    int16_t block_time;

    uint32_t *lace_size = NULL;

    int n, flags, laces = 0;

    uint64_t num;



    /* first byte(s): tracknum */

    if ((n = matroska_ebmlnum_uint(data, size, &num)) < 0) {

        av_log(matroska->ctx, AV_LOG_ERROR, "EBML block data error\n");

        av_free(origdata);

        return res;

    }

    data += n;

    size -= n;



    /* fetch track from num */

    track = matroska_find_track_by_num(matroska, num);

    if (ptrack)  *ptrack = track;

    if (size <= 3 || track < 0 || track >= matroska->num_tracks) {

        av_log(matroska->ctx, AV_LOG_INFO,

               "Invalid stream %d or size %u\n", track, size);

        av_free(origdata);

        return res;

    }

    if(matroska->ctx->streams[ matroska->tracks[track]->stream_index ]->discard >= AVDISCARD_ALL){

        av_free(origdata);

        return res;

    }



    /* block_time (relative to cluster time) */

    block_time = (data[0] << 8) | data[1];

    data += 2;

    size -= 2;

    flags = *data;

    data += 1;

    size -= 1;

    if (is_keyframe == -1)

        is_keyframe = flags & 1 ? PKT_FLAG_KEY : 0;

    switch ((flags & 0x06) >> 1) {

        case 0x0: /* no lacing */

            laces = 1;

            lace_size = av_mallocz(sizeof(int));

            lace_size[0] = size;

            break;



        case 0x1: /* xiph lacing */

        case 0x2: /* fixed-size lacing */

        case 0x3: /* EBML lacing */

            if (size == 0) {

                res = -1;

                break;

            }

            laces = (*data) + 1;

            data += 1;

            size -= 1;

            lace_size = av_mallocz(laces * sizeof(int));



            switch ((flags & 0x06) >> 1) {

                case 0x1: /* xiph lacing */ {

                    uint8_t temp;

                    uint32_t total = 0;

                    for (n = 0; res == 0 && n < laces - 1; n++) {

                        while (1) {

                            if (size == 0) {

                                res = -1;

                                break;

                            }

                            temp = *data;

                            lace_size[n] += temp;

                            data += 1;

                            size -= 1;

                            if (temp != 0xff)

                                break;

                        }

                        total += lace_size[n];

                    }

                    lace_size[n] = size - total;

                    break;

                }



                case 0x2: /* fixed-size lacing */

                    for (n = 0; n < laces; n++)

                        lace_size[n] = size / laces;

                    break;



                case 0x3: /* EBML lacing */ {

                    uint32_t total;

                    n = matroska_ebmlnum_uint(data, size, &num);

                    if (n < 0) {

                        av_log(matroska->ctx, AV_LOG_INFO,

                               "EBML block data error\n");

                        break;

                    }

                    data += n;

                    size -= n;

                    total = lace_size[0] = num;

                    for (n = 1; res == 0 && n < laces - 1; n++) {

                        int64_t snum;

                        int r;

                        r = matroska_ebmlnum_sint (data, size, &snum);

                        if (r < 0) {

                            av_log(matroska->ctx, AV_LOG_INFO,

                                   "EBML block data error\n");

                            break;

                        }

                        data += r;

                        size -= r;

                        lace_size[n] = lace_size[n - 1] + snum;

                        total += lace_size[n];

                    }

                    lace_size[n] = size - total;

                    break;

                }

            }

            break;

    }



    if (res == 0) {

        int real_v = matroska->tracks[track]->flags & MATROSKA_TRACK_REAL_V;

        for (n = 0; n < laces; n++) {

            uint64_t timecode = AV_NOPTS_VALUE;

            int slice, slices = 1;



            if (real_v) {

                slices = *data++ + 1;

                lace_size[n]--;

            }

            if (cluster_time != (uint64_t)-1 && n == 0) {

                if (cluster_time + block_time >= 0)

                    timecode = cluster_time + block_time;

            }

            /* FIXME: duration */



            for (slice=0; slice<slices; slice++) {

                int slice_size, slice_offset = 0;

                if (real_v)

                    slice_offset = rv_offset(data, slice, slices);

                if (slice+1 == slices)

                    slice_size = lace_size[n] - slice_offset;

                else

                    slice_size = rv_offset(data, slice+1, slices) - slice_offset;

                pkt = av_mallocz(sizeof(AVPacket));

                if (ppkt)  *ppkt = pkt;

                /* XXX: prevent data copy... */

                if (av_new_packet(pkt, slice_size) < 0) {

                    res = AVERROR_NOMEM;

                    n = laces-1;

                    break;

                }

                memcpy (pkt->data, data+slice_offset, slice_size);



                if (n == 0)

                    pkt->flags = is_keyframe;

                pkt->stream_index = matroska->tracks[track]->stream_index;



                pkt->pts = timecode;

                pkt->pos = pos;



                if (matroska->tracks[track]->flags & MATROSKA_TRACK_REORDER)

                    matroska_queue_packet_reordered(matroska, pkt, is_bframe);

                else

                    matroska_queue_packet(matroska, pkt);

            }

            data += lace_size[n];

        }

    }



    av_free(lace_size);

    av_free(origdata);

    return res;

}
