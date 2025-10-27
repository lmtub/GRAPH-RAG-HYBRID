static int check(AVIOContext *pb, int64_t pos, uint32_t *ret_header)

{

    int64_t ret = avio_seek(pb, pos, SEEK_SET);

    uint8_t header_buf[4];

    unsigned header;

    MPADecodeHeader sd;

    if (ret < 0)

        return CHECK_SEEK_FAILED;



    ret = avio_read(pb, &header_buf[0], 4);

    if (ret < 0)

        return CHECK_SEEK_FAILED;



    header = AV_RB32(&header_buf[0]);

    if (ff_mpa_check_header(header) < 0)

        return CHECK_WRONG_HEADER;

    if (avpriv_mpegaudio_decode_header(&sd, header) == 1)

        return CHECK_WRONG_HEADER;



    if (ret_header)

        *ret_header = header;

    return sd.frame_size;

}
