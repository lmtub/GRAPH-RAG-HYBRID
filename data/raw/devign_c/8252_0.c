static void RENAME(lumRangeToJpeg)(int16_t *dst, int width)

{

    int i;

    for (i = 0; i < width; i++)

        dst[i] = (FFMIN(dst[i],30189)*19077 - 39057361)>>14;

}
