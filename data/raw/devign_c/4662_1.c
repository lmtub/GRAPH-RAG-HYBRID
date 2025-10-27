static void FUNCC(pred4x4_129_dc)(uint8_t *_src, const uint8_t *topright, int _stride){

    pixel *src = (pixel*)_src;

    int stride = _stride/sizeof(pixel);

    ((pixel4*)(src+0*stride))[0]=

    ((pixel4*)(src+1*stride))[0]=

    ((pixel4*)(src+2*stride))[0]=

    ((pixel4*)(src+3*stride))[0]= PIXEL_SPLAT_X4((1<<(BIT_DEPTH-1))+1);

}
