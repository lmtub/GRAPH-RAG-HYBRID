static unsigned decode_skip_count(GetBitContext* gb)

{

    unsigned value;

    // This function reads a maximum of 23 bits,

    // which is within the padding space

    if (!can_safely_read(gb, 1))

        return -1;

    value = get_bits1(gb);

    if (!value)

        return value;



    value += get_bits(gb, 3);

    if (value != (1 + ((1 << 3) - 1)))

        return value;



    value += get_bits(gb, 7);

    if (value != (1 + ((1 << 3) - 1)) + ((1 << 7) - 1))

        return value;



    return value + get_bits(gb, 12);

}
