static void FUNC(transquant_bypass8x8)(uint8_t *_dst, int16_t *coeffs,

                                       ptrdiff_t stride)

{

    int x, y;

    pixel *dst = (pixel *)_dst;



    stride /= sizeof(pixel);



    for (y = 0; y < 8; y++) {

        for (x = 0; x < 8; x++) {

            dst[x] += *coeffs;

            coeffs++;

        }

        dst += stride;

    }

}
