static int get_packetheader(NUTContext *nut, ByteIOContext *bc, int prefix_length, int calculate_checksum)

{

    int64_t start, size, last_size;

    start= url_ftell(bc) - prefix_length;



    if(start != nut->packet_start + nut->written_packet_size){

        av_log(nut->avf, AV_LOG_ERROR, "get_packetheader called at weird position\n");

        return -1;

    }

    

    if(calculate_checksum)

        init_checksum(bc, update_adler32, 0);



    size= get_v(bc);

    last_size= get_v(bc);

    if(nut->written_packet_size != last_size){

        av_log(nut->avf, AV_LOG_ERROR, "packet size missmatch %d != %lld at %lld\n", nut->written_packet_size, last_size, start);

        return -1;

    }



    nut->last_packet_start = nut->packet_start;

    nut->packet_start = start;

    nut->written_packet_size= size;



    return size;

}
