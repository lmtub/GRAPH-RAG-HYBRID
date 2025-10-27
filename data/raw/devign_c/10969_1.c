static int avi_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    AVIContext *avi = s->priv_data;
    ByteIOContext *pb = s->pb;
    int n, d[8], size;
    offset_t i, sync;
    void* dstr;
    if (ENABLE_DV_DEMUXER && avi->dv_demux) {
        size = dv_get_packet(avi->dv_demux, pkt);
        if (size >= 0)
            return size;
    }
    if(avi->non_interleaved){
        int best_stream_index = 0;
        AVStream *best_st= NULL;
        AVIStream *best_ast;
        int64_t best_ts= INT64_MAX;
        int i;
        for(i=0; i<s->nb_streams; i++){
            AVStream *st = s->streams[i];
            AVIStream *ast = st->priv_data;
            int64_t ts= ast->frame_offset;
            if(ast->sample_size)
                ts /= ast->sample_size;
            ts= av_rescale(ts, AV_TIME_BASE * (int64_t)st->time_base.num, st->time_base.den);
//            av_log(NULL, AV_LOG_DEBUG, "%"PRId64" %d/%d %"PRId64"\n", ts, st->time_base.num, st->time_base.den, ast->frame_offset);
            if(ts < best_ts && st->nb_index_entries){
                best_ts= ts;
                best_st= st;
                best_stream_index= i;
            }
        }
        best_ast = best_st->priv_data;
        best_ts= av_rescale(best_ts, best_st->time_base.den, AV_TIME_BASE * (int64_t)best_st->time_base.num); //FIXME a little ugly
        if(best_ast->remaining)
            i= av_index_search_timestamp(best_st, best_ts, AVSEEK_FLAG_ANY | AVSEEK_FLAG_BACKWARD);
        else
            i= av_index_search_timestamp(best_st, best_ts, AVSEEK_FLAG_ANY);
//        av_log(NULL, AV_LOG_DEBUG, "%d\n", i);
        if(i>=0){
            int64_t pos= best_st->index_entries[i].pos;
            pos += best_ast->packet_size - best_ast->remaining;
            url_fseek(s->pb, pos + 8, SEEK_SET);
//        av_log(NULL, AV_LOG_DEBUG, "pos=%"PRId64"\n", pos);
            assert(best_ast->remaining <= best_ast->packet_size);
            avi->stream_index= best_stream_index;
            if(!best_ast->remaining)
                best_ast->packet_size=
                best_ast->remaining= best_st->index_entries[i].size;
        }
    }
resync:
    if(avi->stream_index >= 0){
        AVStream *st= s->streams[ avi->stream_index ];
        AVIStream *ast= st->priv_data;
        int size;
        if(ast->sample_size <= 1) // minorityreport.AVI block_align=1024 sample_size=1 IMA-ADPCM
            size= INT_MAX;
        else if(ast->sample_size < 32)
            size= 64*ast->sample_size;
        else
            size= ast->sample_size;
        if(size > ast->remaining)
            size= ast->remaining;
        av_get_packet(pb, pkt, size);
        if(ast->has_pal && pkt->data && pkt->size<(unsigned)INT_MAX/2){
            ast->has_pal=0;
            pkt->size += 4*256;
            pkt->data = av_realloc(pkt->data, pkt->size + FF_INPUT_BUFFER_PADDING_SIZE);
            if(pkt->data)
                memcpy(pkt->data + pkt->size - 4*256, ast->pal, 4*256);
        }
        if (ENABLE_DV_DEMUXER && avi->dv_demux) {
            dstr = pkt->destruct;
            size = dv_produce_packet(avi->dv_demux, pkt,
                                    pkt->data, pkt->size);
            pkt->destruct = dstr;
            pkt->flags |= PKT_FLAG_KEY;
        } else {
            /* XXX: how to handle B frames in avi ? */
            pkt->dts = ast->frame_offset;
//                pkt->dts += ast->start;
            if(ast->sample_size)
                pkt->dts /= ast->sample_size;
//av_log(NULL, AV_LOG_DEBUG, "dts:%"PRId64" offset:%"PRId64" %d/%d smpl_siz:%d base:%d st:%d size:%d\n", pkt->dts, ast->frame_offset, ast->scale, ast->rate, ast->sample_size, AV_TIME_BASE, avi->stream_index, size);
            pkt->stream_index = avi->stream_index;
            if (st->codec->codec_type == CODEC_TYPE_VIDEO) {
                AVIndexEntry *e;
                int index;
                assert(st->index_entries);
                index= av_index_search_timestamp(st, pkt->dts, 0);
                e= &st->index_entries[index];
                if(index >= 0 && e->timestamp == ast->frame_offset){
                    if (e->flags & AVINDEX_KEYFRAME)
                        pkt->flags |= PKT_FLAG_KEY;
                }
            } else {
                pkt->flags |= PKT_FLAG_KEY;
            }
            if(ast->sample_size)
                ast->frame_offset += pkt->size;
            else
                ast->frame_offset++;
        }
        ast->remaining -= size;
        if(!ast->remaining){
            avi->stream_index= -1;
            ast->packet_size= 0;
        }
        return size;
    }
    memset(d, -1, sizeof(int)*8);
    for(i=sync=url_ftell(pb); !url_feof(pb); i++) {
        int j;
        for(j=0; j<7; j++)
            d[j]= d[j+1];
        d[7]= get_byte(pb);
        size= d[4] + (d[5]<<8) + (d[6]<<16) + (d[7]<<24);
        if(    d[2] >= '0' && d[2] <= '9'
            && d[3] >= '0' && d[3] <= '9'){
            n= (d[2] - '0') * 10 + (d[3] - '0');
        }else{
            n= 100; //invalid stream id
        }
//av_log(NULL, AV_LOG_DEBUG, "%X %X %X %X %X %X %X %X %"PRId64" %d %d\n", d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], i, size, n);
        if(i + size > avi->fsize || d[0]<0)
            continue;
        //parse ix##
        if(  (d[0] == 'i' && d[1] == 'x' && n < s->nb_streams)
        //parse JUNK
           ||(d[0] == 'J' && d[1] == 'U' && d[2] == 'N' && d[3] == 'K')
           ||(d[0] == 'i' && d[1] == 'd' && d[2] == 'x' && d[3] == '1')){
            url_fskip(pb, size);
//av_log(NULL, AV_LOG_DEBUG, "SKIP\n");
            goto resync;
        }
        if(    d[0] >= '0' && d[0] <= '9'
            && d[1] >= '0' && d[1] <= '9'){
            n= (d[0] - '0') * 10 + (d[1] - '0');
        }else{
            n= 100; //invalid stream id
        }
        //parse ##dc/##wb
        if(n < s->nb_streams){
            AVStream *st;
            AVIStream *ast;
            st = s->streams[n];
            ast = st->priv_data;
            if(s->nb_streams>=2){
                AVStream *st1  = s->streams[1];
                AVIStream *ast1= st1->priv_data;
                //workaround for broken small-file-bug402.avi
                if(   d[2] == 'w' && d[3] == 'b'
                   && n==0
                   && st ->codec->codec_type == CODEC_TYPE_VIDEO
                   && st1->codec->codec_type == CODEC_TYPE_AUDIO
                   && ast->prefix == 'd'*256+'c'
                   && (d[2]*256+d[3] == ast1->prefix || !ast1->prefix_count)
                  ){
                    n=1;
                    st = st1;
                    ast = ast1;
                    av_log(s, AV_LOG_WARNING, "Invalid stream+prefix combination, assuming audio\n");
                }
            }
            if(   (st->discard >= AVDISCARD_DEFAULT && size==0)
               /*|| (st->discard >= AVDISCARD_NONKEY && !(pkt->flags & PKT_FLAG_KEY))*/ //FIXME needs a little reordering
               || st->discard >= AVDISCARD_ALL){
                if(ast->sample_size) ast->frame_offset += pkt->size;
                else                 ast->frame_offset++;
                url_fskip(pb, size);
                goto resync;
            }
            if (d[2] == 'p' && d[3] == 'c' && size<=4*256+4) {
                int k = get_byte(pb);
                int last = (k + get_byte(pb) - 1) & 0xFF;
                get_le16(pb); //flags
                for (; k <= last; k++)
                    ast->pal[k] = get_be32(pb)>>8;// b + (g << 8) + (r << 16);
                ast->has_pal= 1;
                goto resync;
            } else if(   ((ast->prefix_count<5 || sync+9 > i) && d[2]<128 && d[3]<128) ||
                         d[2]*256+d[3] == ast->prefix /*||
                         (d[2] == 'd' && d[3] == 'c') ||
                         (d[2] == 'w' && d[3] == 'b')*/) {
//av_log(NULL, AV_LOG_DEBUG, "OK\n");
                if(d[2]*256+d[3] == ast->prefix)
                    ast->prefix_count++;
                else{
                    ast->prefix= d[2]*256+d[3];
                    ast->prefix_count= 0;
                }
                avi->stream_index= n;
                ast->packet_size= size + 8;
                ast->remaining= size;
                {
                    uint64_t pos= url_ftell(pb) - 8;
                    if(!st->index_entries || !st->nb_index_entries || st->index_entries[st->nb_index_entries - 1].pos < pos){
                        av_add_index_entry(st, pos, ast->frame_offset / FFMAX(1, ast->sample_size), size, 0, AVINDEX_KEYFRAME);
                    }
                }
                goto resync;
            }
        }
    }
}