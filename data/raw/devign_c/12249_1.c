static inline int get_block(GetBitContext *gb, DCTELEM *block, const uint8_t *scan,
                            const uint32_t *quant) {
    int coeff, i, n;
    int8_t ac;
    uint8_t dc = get_bits(gb, 8);
    // block not coded
    if (dc == 255)
    // number of non-zero coefficients
    coeff = get_bits(gb, 6);
    // normally we would only need to clear the (63 - coeff) last values,
    // but since we do not know where they are we just clear the whole block
    memset(block, 0, 64 * sizeof(DCTELEM));
    // 2 bits per coefficient
    while (coeff) {
        ac = get_sbits(gb, 2);
        if (ac == -2)
            break; // continue with more bits
        PUT_COEFF(ac);
    }
    // 4 bits per coefficient
    ALIGN(4);
    if (get_bits_count(gb) + (coeff << 2) >= gb->size_in_bits)
    while (coeff) {
        ac = get_sbits(gb, 4);
        if (ac == -8)
            break; // continue with more bits
        PUT_COEFF(ac);
    }
    // 8 bits per coefficient
    ALIGN(8);
    if (get_bits_count(gb) + (coeff << 3) >= gb->size_in_bits)
    while (coeff) {
        ac = get_sbits(gb, 8);
        PUT_COEFF(ac);
    }
    PUT_COEFF(dc);
    return 1;
}