static void FUNCC(pred8x8l_horizontal)(uint8_t *_src, int has_topleft, int has_topright, int _stride)

{

    pixel *src = (pixel*)_src;

    int stride = _stride/sizeof(pixel);



    PREDICT_8x8_LOAD_LEFT;

#define ROW(y) ((pixel4*)(src+y*stride))[0] =\

               ((pixel4*)(src+y*stride))[1] = PIXEL_SPLAT_X4(l##y)

    ROW(0); ROW(1); ROW(2); ROW(3); ROW(4); ROW(5); ROW(6); ROW(7);

#undef ROW

}
