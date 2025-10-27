uint32_t av_crc(const AVCRC *ctx, uint32_t crc, const uint8_t *buffer, size_t length){

    const uint8_t *end= buffer+length;



#if !CONFIG_SMALL

    if(!ctx[256])

        while(buffer<end-3){

            crc ^= le2me_32(*(const uint32_t*)buffer); buffer+=4;

            crc =  ctx[3*256 + ( crc     &0xFF)]

                  ^ctx[2*256 + ((crc>>8 )&0xFF)]

                  ^ctx[1*256 + ((crc>>16)&0xFF)]

                  ^ctx[0*256 + ((crc>>24)     )];

        }

#endif

    while(buffer<end)

        crc = ctx[((uint8_t)crc) ^ *buffer++] ^ (crc >> 8);



    return crc;

}
