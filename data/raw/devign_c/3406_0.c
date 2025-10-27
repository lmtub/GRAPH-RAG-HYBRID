static uint8_t *ogg_write_vorbiscomment(int offset, int bitexact,

                                        int *header_len, AVDictionary **m, int framing_bit)

{

    const char *vendor = bitexact ? "ffmpeg" : LIBAVFORMAT_IDENT;

    int size;

    uint8_t *p, *p0;



    ff_metadata_conv(m, ff_vorbiscomment_metadata_conv, NULL);



    size = offset + ff_vorbiscomment_length(*m, vendor) + framing_bit;

    p = av_mallocz(size);

    if (!p)

        return NULL;

    p0 = p;



    p += offset;

    ff_vorbiscomment_write(&p, m, vendor);

    if (framing_bit)

        bytestream_put_byte(&p, 1);



    *header_len = size;

    return p0;

}
