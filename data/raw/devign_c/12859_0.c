static void FUNCC(pred8x8l_horizontal_add)(uint8_t *_pix, const int16_t *_block,

                                           ptrdiff_t stride)

{

    int i;

    pixel *pix = (pixel*)_pix;

    const dctcoef *block = (const dctcoef*)_block;

    stride >>= sizeof(pixel)-1;

    for(i=0; i<8; i++){

        pixel v = pix[-1];

        pix[0]= v += block[0];

        pix[1]= v += block[1];

        pix[2]= v += block[2];

        pix[3]= v += block[3];

        pix[4]= v += block[4];

        pix[5]= v += block[5];

        pix[6]= v += block[6];

        pix[7]= v +  block[7];

        pix+= stride;

        block+= 8;

    }

}
