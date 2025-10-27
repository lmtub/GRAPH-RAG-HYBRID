static void RENAME(lumRangeFromJpeg)(int16_t *dst, int width)

{

    int i;

    for (i = 0; i < width; i++)

        dst[i] = (dst[i]*14071 + 33561947)>>14;

}
