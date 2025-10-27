static inline int test_bit(uint32_t *field, int bit)

{

    return (field[bit >> 5] & 1 << (bit & 0x1F)) != 0;

}
