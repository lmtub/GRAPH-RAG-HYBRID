static void svq1_parse_string(GetBitContext *bitbuf, uint8_t *out)

{

    uint8_t seed;

    int i;



    out[0] = get_bits(bitbuf, 8);

    seed   = string_table[out[0]];



    for (i = 1; i <= out[0]; i++) {

        out[i] = get_bits(bitbuf, 8) ^ seed;

        seed   = string_table[out[i] ^ seed];

    }

}
