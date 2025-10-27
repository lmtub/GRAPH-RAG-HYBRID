static int ogg_build_flac_headers(const uint8_t *extradata, int extradata_size,

                                  OGGStreamContext *oggstream, int bitexact)

{

    const char *vendor = bitexact ? "ffmpeg" : LIBAVFORMAT_IDENT;

    uint8_t *p;

    if (extradata_size != 34)

        return -1;

    oggstream->header_len[0] = 51;

    oggstream->header[0] = av_mallocz(51); // per ogg flac specs

    p = oggstream->header[0];

    bytestream_put_byte(&p, 0x7F);

    bytestream_put_buffer(&p, "FLAC", 4);

    bytestream_put_byte(&p, 1); // major version

    bytestream_put_byte(&p, 0); // minor version

    bytestream_put_be16(&p, 1); // headers packets without this one

    bytestream_put_buffer(&p, "fLaC", 4);

    bytestream_put_byte(&p, 0x00); // streaminfo

    bytestream_put_be24(&p, 34);

    bytestream_put_buffer(&p, extradata, 34);

    oggstream->header_len[1] = 1+3+4+strlen(vendor)+4;

    oggstream->header[1] = av_mallocz(oggstream->header_len[1]);

    p = oggstream->header[1];

    bytestream_put_byte(&p, 0x84); // last metadata block and vorbis comment

    bytestream_put_be24(&p, oggstream->header_len[1] - 4);

    bytestream_put_le32(&p, strlen(vendor));

    bytestream_put_buffer(&p, vendor, strlen(vendor));

    bytestream_put_le32(&p, 0); // user comment list length

    return 0;

}
