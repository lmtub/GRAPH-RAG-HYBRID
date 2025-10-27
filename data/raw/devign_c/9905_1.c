static void FUNCC(pred16x16_horizontal)(uint8_t *_src, int stride){

    int i;

    pixel *src = (pixel*)_src;

    stride /= sizeof(pixel);



    for(i=0; i<16; i++){

        ((pixel4*)(src+i*stride))[0] =

        ((pixel4*)(src+i*stride))[1] =

        ((pixel4*)(src+i*stride))[2] =

        ((pixel4*)(src+i*stride))[3] = PIXEL_SPLAT_X4(src[-1+i*stride]);

    }

}
