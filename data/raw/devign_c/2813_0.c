static inline void bit_copy(PutBitContext *pb, GetBitContext *gb)

{

    int bits_left = get_bits_left(gb);

    while (bits_left >= 16) {

        put_bits(pb, 16, get_bits(gb, 16));

        bits_left -= 16;

    }

    if (bits_left > 0) {

        put_bits(pb, bits_left, get_bits(gb, bits_left));

    }

}
