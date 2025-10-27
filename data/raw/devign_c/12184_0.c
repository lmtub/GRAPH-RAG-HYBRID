static int ty_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    TYDemuxContext *ty = s->priv_data;

    AVIOContext *pb = s->pb;

    TyRecHdr *rec;

    int64_t rec_size = 0;

    int ret = 0;



    if (avio_feof(pb))

        return AVERROR_EOF;



    while (ret <= 0) {

        if (ty->first_chunk || ty->cur_rec >= ty->num_recs) {

            if (get_chunk(s) < 0 || ty->num_recs == 0)

                return AVERROR_EOF;

        }



        rec = &ty->rec_hdrs[ty->cur_rec];

        rec_size = rec->rec_size;

        ty->cur_rec++;



        if (rec_size <= 0)

            continue;



        if (ty->cur_chunk_pos + rec->rec_size > CHUNK_SIZE)

            return AVERROR_INVALIDDATA;



        if (avio_feof(pb))

            return AVERROR_EOF;



        switch (rec->rec_type) {

        case VIDEO_ID:

            ret = demux_video(s, rec, pkt);

            break;

        case AUDIO_ID:

            ret = demux_audio(s, rec, pkt);

            break;

        default:

            ff_dlog(s, "Invalid record type 0x%02x\n", rec->rec_type);

        case 0x01:

        case 0x02:

        case 0x03: /* TiVo data services */

        case 0x05: /* unknown, but seen regularly */

            ty->cur_chunk_pos += rec->rec_size;

            break;

        }

    }



    return 0;

}
