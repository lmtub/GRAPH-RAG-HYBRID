static uint8_t lag_calc_zero_run(int8_t x)

{

    return (x << 1) ^ (x >> 7);

}
