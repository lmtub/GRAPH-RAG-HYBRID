static void FUNCC(pred8x8_horizontal)(uint8_t *_src, int stride){

    int i;

    pixel *src = (pixel*)_src;

    stride /= sizeof(pixel);



    for(i=0; i<8; i++){

        ((pixel4*)(src+i*stride))[0]=

        ((pixel4*)(src+i*stride))[1]= PIXEL_SPLAT_X4(src[-1+i*stride]);

    }

}
