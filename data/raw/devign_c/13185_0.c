static inline int decode_bytes(const uint8_t* inbuffer, uint8_t* out, int bytes){

    int i, off;

    uint32_t c;

    const uint32_t* buf;

    uint32_t* obuf = (uint32_t*) out;

    /* FIXME: 64 bit platforms would be able to do 64 bits at a time.

     * I'm too lazy though, should be something like

     * for(i=0 ; i<bitamount/64 ; i++)

     *     (int64_t)out[i] = 0x37c511f237c511f2^av_be2ne64(int64_t)in[i]);

     * Buffer alignment needs to be checked. */



    off = (intptr_t)inbuffer & 3;

    buf = (const uint32_t*) (inbuffer - off);

    c = av_be2ne32((0x37c511f2 >> (off*8)) | (0x37c511f2 << (32-(off*8))));

    bytes += 3 + off;

    for (i = 0; i < bytes/4; i++)

        obuf[i] = c ^ buf[i];



    return off;

}
