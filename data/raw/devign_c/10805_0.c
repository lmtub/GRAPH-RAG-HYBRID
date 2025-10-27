int url_open_dyn_packet_buf(AVIOContext **s, int max_packet_size)

{

    if (max_packet_size <= 0)

        return -1;

    return url_open_dyn_buf_internal(s, max_packet_size);

}
