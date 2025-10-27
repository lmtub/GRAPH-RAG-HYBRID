static inline float to_float(uint8_t exp, int16_t mantissa)

{

    return ((float) (mantissa * scale_factors[exp]));

}
