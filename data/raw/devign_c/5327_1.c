uint8_t ff_mlp_calculate_parity(const uint8_t *buf, unsigned int buf_size)

{

    uint32_t scratch = 0;

    const uint8_t *buf_end = buf + buf_size;





    for (; buf < buf_end - 3; buf += 4)

        scratch ^= *((const uint32_t*)buf);



    scratch = xor_32_to_8(scratch);



    for (; buf < buf_end; buf++)




    return scratch;

}