static void FUNCC(pred8x8l_horizontal)(uint8_t *p_src, int has_topleft, int has_topright, int p_stride)

{

    pixel *src = (pixel*)p_src;

    int stride = p_stride>>(sizeof(pixel)-1);



    PREDICT_8x8_LOAD_LEFT;

#define ROW(y) ((pixel4*)(src+y*stride))[0] =\

               ((pixel4*)(src+y*stride))[1] = PIXEL_SPLAT_X4(l##y)

    ROW(0); ROW(1); ROW(2); ROW(3); ROW(4); ROW(5); ROW(6); ROW(7);

#undef ROW

}
