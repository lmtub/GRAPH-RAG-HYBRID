static int decode_bytes(const uint8_t *input, uint8_t *out, int bytes)

{

    int i, off;

    uint32_t c;

    const uint32_t *buf;

    uint32_t *output = (uint32_t *)out;



    off = (intptr_t)input & 3;

    buf = (const uint32_t *)(input - off);

    c   = av_be2ne32((0x537F6103 >> (off * 8)) | (0x537F6103 << (32 - (off * 8))));

    bytes += 3 + off;

    for (i = 0; i < bytes / 4; i++)

        output[i] = c ^ buf[i];



    if (off)

        avpriv_request_sample(NULL, "Offset of %d", off);



    return off;

}
