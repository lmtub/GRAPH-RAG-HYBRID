static void decodeplane32(uint32_t *dst, const uint8_t *const buf, int buf_size, int bps, int plane)

{

    GetBitContext gb;

    int i, b;

    init_get_bits(&gb, buf, buf_size * 8);

    for(i = 0; i < (buf_size * 8 + bps - 1) / bps; i++) {

        for (b = 0; b < bps; b++) {

            dst[ i*bps + b ] |= get_bits1(&gb) << plane;

        }

    }

}
