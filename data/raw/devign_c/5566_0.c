static int cow_test_bit(int64_t bitnum, const uint8_t *bitmap)

{

    return (bitmap[bitnum / 8] & (1 << (bitnum & 7))) != 0;

}
