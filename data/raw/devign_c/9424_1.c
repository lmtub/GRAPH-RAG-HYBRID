static void decode_rowskip(uint8_t* plane, int width, int height, int stride, VC9Context *v){

    int x, y;

    GetBitContext *gb = &v->s.gb;



    for (y=0; y<height; y++){

        if (!get_bits(gb, 1)) //rowskip

            memset(plane, 0, width);

        else

            for (x=0; x<width; x++) 

                plane[x] = get_bits(gb, 1);

        plane += stride;

    }

}
