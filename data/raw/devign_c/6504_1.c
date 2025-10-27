matroska_parse_block(MatroskaDemuxContext *matroska, uint8_t *data, int size,

                     int64_t pos, uint64_t cluster_time, uint64_t duration,

                     int is_keyframe, int is_bframe)

{

    int res = 0;

    int track;

    AVStream *st;

    AVPacket *pkt;

    uint8_t *origdata = data;

    int16_t block_time;

    uint32_t *lace_size = NULL;

    int n, flags, laces = 0;

    uint64_t num;

    int stream_index;



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

    if (size <= 3 || track < 0 || track >= matroska->num_tracks) {

        av_log(matroska->ctx, AV_LOG_INFO,

               "Invalid stream %d or size %u\n", track, size);

        av_free(origdata);

        return res;

    }

    stream_index = matroska->tracks[track]->stream_index;

    if (stream_index < 0 || stream_index >= matroska->ctx->nb_streams) {

        av_free(origdata);

        return res;

    }

    st = matroska->ctx->streams[stream_index];

    if (st->discard >= AVDISCARD_ALL) {

        av_free(origdata);

        return res;

    }

    if (duration == AV_NOPTS_VALUE)

        duration = matroska->tracks[track]->default_duration / matroska->time_scale;



    /* block_time (relative to cluster time) */

    block_time = AV_RB16(data);

    data += 2;

    flags = *data++;

    size -= 3;

    if (is_keyframe == -1)

        is_keyframe = flags & 0x80 ? PKT_FLAG_KEY : 0;



    if (matroska->skip_to_keyframe) {

        if (!is_keyframe || st != matroska->skip_to_stream) {

            av_free(origdata);

            return res;

        }

        matroska->skip_to_keyframe = 0;

    }



    switch ((flags & 0x06) >> 1) {

        case 0x0: /* no lacing */

            laces = 1;

            lace_size = av_mallocz(sizeof(int));

            lace_size[0] = size;

            break;



        case 0x1: /* xiph lacing */

        case 0x2: /* fixed-size lacing */

        case 0x3: /* EBML lacing */

            assert(size>0); // size <=3 is checked before size-=3 above

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

        uint64_t timecode = AV_NOPTS_VALUE;



        if (cluster_time != (uint64_t)-1

            && (block_time >= 0 || cluster_time >= -block_time))

            timecode = cluster_time + block_time;



        for (n = 0; n < laces; n++) {

            if (st->codec->codec_id == CODEC_ID_RA_288 ||

                st->codec->codec_id == CODEC_ID_COOK ||

                st->codec->codec_id == CODEC_ID_ATRAC3) {

                MatroskaAudioTrack *audiotrack = (MatroskaAudioTrack *)matroska->tracks[track];

                int a = st->codec->block_align;

                int sps = audiotrack->sub_packet_size;

                int cfs = audiotrack->coded_framesize;

                int h = audiotrack->sub_packet_h;

                int y = audiotrack->sub_packet_cnt;

                int w = audiotrack->frame_size;

                int x;



                if (!audiotrack->pkt_cnt) {

                    if (st->codec->codec_id == CODEC_ID_RA_288)

                        for (x=0; x<h/2; x++)

                            memcpy(audiotrack->buf+x*2*w+y*cfs,

                                   data+x*cfs, cfs);

                    else

                        for (x=0; x<w/sps; x++)

                            memcpy(audiotrack->buf+sps*(h*x+((h+1)/2)*(y&1)+(y>>1)), data+x*sps, sps);



                    if (++audiotrack->sub_packet_cnt >= h) {

                        audiotrack->sub_packet_cnt = 0;

                        audiotrack->pkt_cnt = h*w / a;

                    }

                }

                while (audiotrack->pkt_cnt) {

                    pkt = av_mallocz(sizeof(AVPacket));

                    av_new_packet(pkt, a);

                    memcpy(pkt->data, audiotrack->buf

                           + a * (h*w / a - audiotrack->pkt_cnt--), a);

                    pkt->pos = pos;

                    pkt->stream_index = stream_index;

                    matroska_queue_packet(matroska, pkt);

                }

            } else {

                int result, offset = 0, ilen, olen, pkt_size = lace_size[n];

                uint8_t *pkt_data = data;



                if (matroska->tracks[track]->encoding_scope & 1) {

                    switch (matroska->tracks[track]->encoding_algo) {

                    case MATROSKA_TRACK_ENCODING_COMP_HEADERSTRIP:

                        offset = matroska->tracks[track]->encoding_settings_len;

                        break;

                    case MATROSKA_TRACK_ENCODING_COMP_LZO:

                        pkt_data = NULL;

                        do {

                            ilen = lace_size[n];

                            olen = pkt_size *= 3;

                            pkt_data = av_realloc(pkt_data,

                                                  pkt_size+LZO_OUTPUT_PADDING);

                            result = lzo1x_decode(pkt_data, &olen, data, &ilen);

                        } while (result==LZO_OUTPUT_FULL && pkt_size<10000000);

                        if (result) {

                            av_free(pkt_data);

                            continue;

                        }

                        pkt_size -= olen;

                        break;

#ifdef CONFIG_ZLIB

                    case MATROSKA_TRACK_ENCODING_COMP_ZLIB: {

                        z_stream zstream = {0};

                        pkt_data = NULL;

                        if (inflateInit(&zstream) != Z_OK)

                            continue;

                        zstream.next_in = data;

                        zstream.avail_in = lace_size[n];

                        do {

                            pkt_size *= 3;

                            pkt_data = av_realloc(pkt_data, pkt_size);

                            zstream.avail_out = pkt_size - zstream.total_out;

                            zstream.next_out = pkt_data + zstream.total_out;

                            result = inflate(&zstream, Z_NO_FLUSH);

                        } while (result==Z_OK && pkt_size<10000000);

                        pkt_size = zstream.total_out;

                        inflateEnd(&zstream);

                        if (result != Z_STREAM_END) {

                            av_free(pkt_data);

                            continue;

                        }

                        break;

                    }

#endif

#ifdef CONFIG_BZLIB

                    case MATROSKA_TRACK_ENCODING_COMP_BZLIB: {

                        bz_stream bzstream = {0};

                        pkt_data = NULL;

                        if (BZ2_bzDecompressInit(&bzstream, 0, 0) != BZ_OK)

                            continue;

                        bzstream.next_in = data;

                        bzstream.avail_in = lace_size[n];

                        do {

                            pkt_size *= 3;

                            pkt_data = av_realloc(pkt_data, pkt_size);

                            bzstream.avail_out = pkt_size - bzstream.total_out_lo32;

                            bzstream.next_out = pkt_data + bzstream.total_out_lo32;

                            result = BZ2_bzDecompress(&bzstream);

                        } while (result==BZ_OK && pkt_size<10000000);

                        pkt_size = bzstream.total_out_lo32;

                        BZ2_bzDecompressEnd(&bzstream);

                        if (result != BZ_STREAM_END) {

                            av_free(pkt_data);

                            continue;

                        }

                        break;

                    }

#endif

                    }

                }



                pkt = av_mallocz(sizeof(AVPacket));

                /* XXX: prevent data copy... */

                if (av_new_packet(pkt, pkt_size+offset) < 0) {


                    res = AVERROR(ENOMEM);

                    n = laces-1;

                    break;

                }

                if (offset)

                    memcpy (pkt->data, matroska->tracks[track]->encoding_settings, offset);

                memcpy (pkt->data+offset, pkt_data, pkt_size);



                if (n == 0)

                    pkt->flags = is_keyframe;

                pkt->stream_index = stream_index;



                pkt->pts = timecode;

                pkt->pos = pos;

                pkt->duration = duration;



                matroska_queue_packet(matroska, pkt);

            }



            if (timecode != AV_NOPTS_VALUE)

                timecode = duration ? timecode + duration : AV_NOPTS_VALUE;

            data += lace_size[n];

        }

    }



    av_free(lace_size);

    av_free(origdata);

    return res;

}