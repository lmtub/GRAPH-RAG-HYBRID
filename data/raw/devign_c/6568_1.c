rgb48funcs(rgb, LE, PIX_FMT_RGB48LE)

rgb48funcs(rgb, BE, PIX_FMT_RGB48BE)

rgb48funcs(bgr, LE, PIX_FMT_BGR48LE)

rgb48funcs(bgr, BE, PIX_FMT_BGR48BE)



#define input_pixel(i) ((origin == PIX_FMT_RGBA || origin == PIX_FMT_BGRA || \

                         origin == PIX_FMT_ARGB || origin == PIX_FMT_ABGR) ? AV_RN32A(&src[(i)*4]) : \

                        (isBE(origin) ? AV_RB16(&src[(i)*2]) : AV_RL16(&src[(i)*2])))



static av_always_inline void

rgb16_32ToY_c_template(uint8_t *dst, const uint8_t *src,

                       int width, enum PixelFormat origin,

                       int shr,   int shg,   int shb, int shp,

                       int maskr, int maskg, int maskb,

                       int rsh,   int gsh,   int bsh, int S)

{

    const int ry = RY << rsh, gy = GY << gsh, by = BY << bsh,

              rnd = 33 << (S - 1);

    int i;



    for (i = 0; i < width; i++) {

        int px = input_pixel(i) >> shp;

        int b = (px & maskb) >> shb;

        int g = (px & maskg) >> shg;

        int r = (px & maskr) >> shr;



        dst[i] = (ry * r + gy * g + by * b + rnd) >> S;

    }

}
