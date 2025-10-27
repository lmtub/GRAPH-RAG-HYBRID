static int ff_asf_parse_packet(AVFormatContext *s, ByteIOContext *pb, AVPacket *pkt)

{

    ASFContext *asf = s->priv_data;

    ASFStream *asf_st = 0;

    for (;;) {

        if(url_feof(pb))

            return AVERROR_EOF;

        if (asf->packet_size_left < FRAME_HEADER_SIZE

            || asf->packet_segments < 1) {

            //asf->packet_size_left <= asf->packet_padsize) {

            int ret = asf->packet_size_left + asf->packet_padsize;

            //printf("PacketLeftSize:%d  Pad:%d Pos:%"PRId64"\n", asf->packet_size_left, asf->packet_padsize, url_ftell(pb));

            assert(ret>=0);

            /* fail safe */

            url_fskip(pb, ret);



            asf->packet_pos= url_ftell(pb);

            if (asf->data_object_size != (uint64_t)-1 &&

                (asf->packet_pos - asf->data_object_offset >= asf->data_object_size))

                return AVERROR_EOF; /* Do not exceed the size of the data object */

            return 1;

        }

        if (asf->packet_time_start == 0) {

            if(asf_read_frame_header(s, pb) < 0){

                asf->packet_segments= 0;

                continue;

            }

            if (asf->stream_index < 0

                || s->streams[asf->stream_index]->discard >= AVDISCARD_ALL

                || (!asf->packet_key_frame && s->streams[asf->stream_index]->discard >= AVDISCARD_NONKEY)

                ) {

                asf->packet_time_start = 0;

                /* unhandled packet (should not happen) */

                url_fskip(pb, asf->packet_frag_size);

                asf->packet_size_left -= asf->packet_frag_size;

                if(asf->stream_index < 0)

                    av_log(s, AV_LOG_ERROR, "ff asf skip %d (unknown stream)\n", asf->packet_frag_size);

                continue;

            }

            asf->asf_st = s->streams[asf->stream_index]->priv_data;

        }

        asf_st = asf->asf_st;



        if (asf->packet_replic_size == 1) {

            // frag_offset is here used as the beginning timestamp

            asf->packet_frag_timestamp = asf->packet_time_start;

            asf->packet_time_start += asf->packet_time_delta;

            asf->packet_obj_size = asf->packet_frag_size = get_byte(pb);

            asf->packet_size_left--;

            asf->packet_multi_size--;

            if (asf->packet_multi_size < asf->packet_obj_size)

            {

                asf->packet_time_start = 0;

                url_fskip(pb, asf->packet_multi_size);

                asf->packet_size_left -= asf->packet_multi_size;

                continue;

            }

            asf->packet_multi_size -= asf->packet_obj_size;

            //printf("COMPRESS size  %d  %d  %d   ms:%d\n", asf->packet_obj_size, asf->packet_frag_timestamp, asf->packet_size_left, asf->packet_multi_size);

        }

        if(   /*asf->packet_frag_size == asf->packet_obj_size*/

              asf_st->frag_offset + asf->packet_frag_size <= asf_st->pkt.size

           && asf_st->frag_offset + asf->packet_frag_size > asf->packet_obj_size){

            av_log(s, AV_LOG_INFO, "ignoring invalid packet_obj_size (%d %d %d %d)\n",

                asf_st->frag_offset, asf->packet_frag_size,

                asf->packet_obj_size, asf_st->pkt.size);

            asf->packet_obj_size= asf_st->pkt.size;

        }



        if (   asf_st->pkt.size != asf->packet_obj_size

            || asf_st->frag_offset + asf->packet_frag_size > asf_st->pkt.size) { //FIXME is this condition sufficient?

            if(asf_st->pkt.data){

                av_log(s, AV_LOG_INFO, "freeing incomplete packet size %d, new %d\n", asf_st->pkt.size, asf->packet_obj_size);

                asf_st->frag_offset = 0;

                av_free_packet(&asf_st->pkt);

            }

            /* new packet */

            av_new_packet(&asf_st->pkt, asf->packet_obj_size);

            asf_st->seq = asf->packet_seq;

            asf_st->pkt.dts = asf->packet_frag_timestamp;

            asf_st->pkt.stream_index = asf->stream_index;

            asf_st->pkt.pos =

            asf_st->packet_pos= asf->packet_pos;

//printf("new packet: stream:%d key:%d packet_key:%d audio:%d size:%d\n",

//asf->stream_index, asf->packet_key_frame, asf_st->pkt.flags & AV_PKT_FLAG_KEY,

//s->streams[asf->stream_index]->codec->codec_type == AVMEDIA_TYPE_AUDIO, asf->packet_obj_size);

            if (s->streams[asf->stream_index]->codec->codec_type == AVMEDIA_TYPE_AUDIO)

                asf->packet_key_frame = 1;

            if (asf->packet_key_frame)

                asf_st->pkt.flags |= AV_PKT_FLAG_KEY;

        }



        /* read data */

        //printf("READ PACKET s:%d  os:%d  o:%d,%d  l:%d   DATA:%p\n",

        //       s->packet_size, asf_st->pkt.size, asf->packet_frag_offset,

        //       asf_st->frag_offset, asf->packet_frag_size, asf_st->pkt.data);

        asf->packet_size_left -= asf->packet_frag_size;

        if (asf->packet_size_left < 0)

            continue;



        if(   asf->packet_frag_offset >= asf_st->pkt.size

           || asf->packet_frag_size > asf_st->pkt.size - asf->packet_frag_offset){

            av_log(s, AV_LOG_ERROR, "packet fragment position invalid %u,%u not in %u\n",

                asf->packet_frag_offset, asf->packet_frag_size, asf_st->pkt.size);

            continue;

        }



        get_buffer(pb, asf_st->pkt.data + asf->packet_frag_offset,

                   asf->packet_frag_size);

        if (s->key && s->keylen == 20)

            ff_asfcrypt_dec(s->key, asf_st->pkt.data + asf->packet_frag_offset,

                            asf->packet_frag_size);

        asf_st->frag_offset += asf->packet_frag_size;

        /* test if whole packet is read */

        if (asf_st->frag_offset == asf_st->pkt.size) {

            //workaround for macroshit radio DVR-MS files

            if(   s->streams[asf->stream_index]->codec->codec_id == CODEC_ID_MPEG2VIDEO

               && asf_st->pkt.size > 100){

                int i;

                for(i=0; i<asf_st->pkt.size && !asf_st->pkt.data[i]; i++);

                if(i == asf_st->pkt.size){

                    av_log(s, AV_LOG_DEBUG, "discarding ms fart\n");

                    asf_st->frag_offset = 0;

                    av_free_packet(&asf_st->pkt);

                    continue;

                }

            }



            /* return packet */

            if (asf_st->ds_span > 1) {

              if(asf_st->pkt.size != asf_st->ds_packet_size * asf_st->ds_span){

                    av_log(s, AV_LOG_ERROR, "pkt.size != ds_packet_size * ds_span (%d %d %d)\n", asf_st->pkt.size, asf_st->ds_packet_size, asf_st->ds_span);

              }else{

                /* packet descrambling */

                uint8_t *newdata = av_malloc(asf_st->pkt.size + FF_INPUT_BUFFER_PADDING_SIZE);

                if (newdata) {

                    int offset = 0;

                    memset(newdata + asf_st->pkt.size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

                    while (offset < asf_st->pkt.size) {

                        int off = offset / asf_st->ds_chunk_size;

                        int row = off / asf_st->ds_span;

                        int col = off % asf_st->ds_span;

                        int idx = row + col * asf_st->ds_packet_size / asf_st->ds_chunk_size;

                        //printf("off:%d  row:%d  col:%d  idx:%d\n", off, row, col, idx);



                        assert(offset + asf_st->ds_chunk_size <= asf_st->pkt.size);

                        assert(idx+1 <= asf_st->pkt.size / asf_st->ds_chunk_size);

                        memcpy(newdata + offset,

                               asf_st->pkt.data + idx * asf_st->ds_chunk_size,

                               asf_st->ds_chunk_size);

                        offset += asf_st->ds_chunk_size;

                    }

                    av_free(asf_st->pkt.data);

                    asf_st->pkt.data = newdata;

                }

              }

            }

            asf_st->frag_offset = 0;

            *pkt= asf_st->pkt;

            //printf("packet %d %d\n", asf_st->pkt.size, asf->packet_frag_size);

            asf_st->pkt.size = 0;

            asf_st->pkt.data = 0;

            break; // packet completed

        }

    }

    return 0;

}
