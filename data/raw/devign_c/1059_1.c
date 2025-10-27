static int check_checksum(ByteIOContext *bc){

    unsigned long checksum= get_checksum(bc);

//    return checksum != get_be32(bc);



    av_log(NULL, AV_LOG_ERROR, "%08X %08X\n", checksum, (int)get_be32(bc));



    return 0;

}
