static void FUNCC(pred8x8_dc)(uint8_t *_src, int stride){

    int i;

    int dc0, dc1, dc2;

    pixel4 dc0splat, dc1splat, dc2splat, dc3splat;

    pixel *src = (pixel*)_src;

    stride /= sizeof(pixel);



    dc0=dc1=dc2=0;

    for(i=0;i<4; i++){

        dc0+= src[-1+i*stride] + src[i-stride];

        dc1+= src[4+i-stride];

        dc2+= src[-1+(i+4)*stride];

    }

    dc0splat = PIXEL_SPLAT_X4((dc0 + 4)>>3);

    dc1splat = PIXEL_SPLAT_X4((dc1 + 2)>>2);

    dc2splat = PIXEL_SPLAT_X4((dc2 + 2)>>2);

    dc3splat = PIXEL_SPLAT_X4((dc1 + dc2 + 4)>>3);



    for(i=0; i<4; i++){

        ((pixel4*)(src+i*stride))[0]= dc0splat;

        ((pixel4*)(src+i*stride))[1]= dc1splat;

    }

    for(i=4; i<8; i++){

        ((pixel4*)(src+i*stride))[0]= dc2splat;

        ((pixel4*)(src+i*stride))[1]= dc3splat;

    }

}
