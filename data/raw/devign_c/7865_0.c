static int avi_read_idx1(AVFormatContext *s, int size)

{

    AVIContext *avi = s->priv_data;

    AVIOContext *pb = s->pb;

    int nb_index_entries, i;

    AVStream *st;

    AVIStream *ast;

    unsigned int index, tag, flags, pos, len, first_packet = 1;

    unsigned last_pos= -1;

    int64_t idx1_pos, first_packet_pos = 0, data_offset = 0;



    nb_index_entries = size / 16;

    if (nb_index_entries <= 0)

        return -1;



    idx1_pos = avio_tell(pb);

    avio_seek(pb, avi->movi_list+4, SEEK_SET);

    if (avi_sync(s, 1) == 0) {

        first_packet_pos = avio_tell(pb) - 8;

    }

    avi->stream_index = -1;

    avio_seek(pb, idx1_pos, SEEK_SET);



    /* Read the entries and sort them in each stream component. */

    for(i = 0; i < nb_index_entries; i++) {

        tag = avio_rl32(pb);

        flags = avio_rl32(pb);

        pos = avio_rl32(pb);

        len = avio_rl32(pb);

        av_dlog(s, "%d: tag=0x%x flags=0x%x pos=0x%x len=%d/",

                i, tag, flags, pos, len);



        index = ((tag & 0xff) - '0') * 10;

        index += ((tag >> 8) & 0xff) - '0';

        if (index >= s->nb_streams)

            continue;

        st = s->streams[index];

        ast = st->priv_data;



        if(first_packet && first_packet_pos && len) {

            data_offset = first_packet_pos - pos;

            first_packet = 0;

        }

        pos += data_offset;



        av_dlog(s, "%d cum_len=%"PRId64"\n", len, ast->cum_len);



        if(pb->eof_reached)

            return -1;



        if(last_pos == pos)

            avi->non_interleaved= 1;

        else if(len || !ast->sample_size)

            av_add_index_entry(st, pos, ast->cum_len, len, 0, (flags&AVIIF_INDEX) ? AVINDEX_KEYFRAME : 0);

        ast->cum_len += get_duration(ast, len);

        last_pos= pos;

    }

    return 0;

}
