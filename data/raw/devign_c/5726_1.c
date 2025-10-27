static void skip_data_stream_element(GetBitContext *gb)

{

    int byte_align = get_bits1(gb);

    int count = get_bits(gb, 8);

    if (count == 255)

        count += get_bits(gb, 8);

    if (byte_align)

        align_get_bits(gb);

    skip_bits_long(gb, 8 * count);

}
