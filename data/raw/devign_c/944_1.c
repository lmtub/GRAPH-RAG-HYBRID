ff_rm_parse_packet (AVFormatContext *s, AVIOContext *pb,

                    AVStream *st, RMStream *ast, int len, AVPacket *pkt,

                    int *seq, int flags, int64_t timestamp)

{

    RMDemuxContext *rm = s->priv_data;

    int ret;



    if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

        rm->current_stream= st->id;

        ret = rm_assemble_video_frame(s, pb, rm, ast, pkt, len, seq, &timestamp);

        if(ret)

            return ret < 0 ? ret : -1; //got partial frame or error

    } else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

        if ((ast->deint_id == DEINT_ID_GENR) ||

            (ast->deint_id == DEINT_ID_INT4) ||

            (ast->deint_id == DEINT_ID_SIPR)) {

            int x;

            int sps = ast->sub_packet_size;

            int cfs = ast->coded_framesize;

            int h = ast->sub_packet_h;

            int y = ast->sub_packet_cnt;

            int w = ast->audio_framesize;



            if (flags & 2)

                y = ast->sub_packet_cnt = 0;

            if (!y)

                ast->audiotimestamp = timestamp;



            switch (ast->deint_id) {

                case DEINT_ID_INT4:

                    for (x = 0; x < h/2; x++)

                        avio_read(pb, ast->pkt.data+x*2*w+y*cfs, cfs);

                    break;

                case DEINT_ID_GENR:

                    for (x = 0; x < w/sps; x++)

                        avio_read(pb, ast->pkt.data+sps*(h*x+((h+1)/2)*(y&1)+(y>>1)), sps);

                    break;

                case DEINT_ID_SIPR:

                    avio_read(pb, ast->pkt.data + y * w, w);

                    break;

            }



            if (++(ast->sub_packet_cnt) < h)

                return -1;

            if (ast->deint_id == DEINT_ID_SIPR)

                ff_rm_reorder_sipr_data(ast->pkt.data, h, w);



             ast->sub_packet_cnt = 0;

             rm->audio_stream_num = st->index;

             rm->audio_pkt_cnt = h * w / st->codec->block_align;

        } else if ((ast->deint_id == DEINT_ID_VBRF) ||

                   (ast->deint_id == DEINT_ID_VBRS)) {

            int x;

            rm->audio_stream_num = st->index;

            ast->sub_packet_cnt = (avio_rb16(pb) & 0xf0) >> 4;

            if (ast->sub_packet_cnt) {

                for (x = 0; x < ast->sub_packet_cnt; x++)

                    ast->sub_packet_lengths[x] = avio_rb16(pb);

                rm->audio_pkt_cnt = ast->sub_packet_cnt;

                ast->audiotimestamp = timestamp;

            } else

                return -1;

        } else {

            av_get_packet(pb, pkt, len);

            rm_ac3_swap_bytes(st, pkt);

        }

    } else

        av_get_packet(pb, pkt, len);



    pkt->stream_index = st->index;



#if 0

    if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

        if(st->codec->codec_id == AV_CODEC_ID_RV20){

            int seq= 128*(pkt->data[2]&0x7F) + (pkt->data[3]>>1);

            av_log(s, AV_LOG_DEBUG, "%d %"PRId64" %d\n", *timestamp, *timestamp*512LL/25, seq);



            seq |= (timestamp&~0x3FFF);

            if(seq - timestamp >  0x2000) seq -= 0x4000;

            if(seq - timestamp < -0x2000) seq += 0x4000;

        }

    }

#endif



    pkt->pts = timestamp;

    if (flags & 2)

        pkt->flags |= AV_PKT_FLAG_KEY;



    return st->codec->codec_type == AVMEDIA_TYPE_AUDIO ? rm->audio_pkt_cnt : 0;

}
