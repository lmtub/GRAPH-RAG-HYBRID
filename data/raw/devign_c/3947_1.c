static void FUNCC(pred8x8_top_dc)(uint8_t *_src, int stride){

    int i;

    int dc0, dc1;

    pixel4 dc0splat, dc1splat;

    pixel *src = (pixel*)_src;

    stride /= sizeof(pixel);



    dc0=dc1=0;

    for(i=0;i<4; i++){

        dc0+= src[i-stride];

        dc1+= src[4+i-stride];

    }

    dc0splat = PIXEL_SPLAT_X4((dc0 + 2)>>2);

    dc1splat = PIXEL_SPLAT_X4((dc1 + 2)>>2);



    for(i=0; i<4; i++){

        ((pixel4*)(src+i*stride))[0]= dc0splat;

        ((pixel4*)(src+i*stride))[1]= dc1splat;

    }

    for(i=4; i<8; i++){

        ((pixel4*)(src+i*stride))[0]= dc0splat;

        ((pixel4*)(src+i*stride))[1]= dc1splat;

    }

}
