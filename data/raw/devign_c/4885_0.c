static inline unsigned char gif_clut_index(uint8_t r, uint8_t g, uint8_t b)

{

    return ((((r)/47)%6)*6*6+(((g)/47)%6)*6+(((b)/47)%6));

}
