static void vc1_put_block(VC1Context *v, DCTELEM block[6][64])

{

    uint8_t *Y;

    int ys, us, vs;

    DSPContext *dsp = &v->s.dsp;



    if(v->rangeredfrm) {

        int i, j, k;

        for(k = 0; k < 6; k++)

            for(j = 0; j < 8; j++)

                for(i = 0; i < 8; i++)

                    block[k][i + j*8] = (block[k][i + j*8] - 64) << 1;



    }

    ys = v->s.current_picture.linesize[0];

    us = v->s.current_picture.linesize[1];

    vs = v->s.current_picture.linesize[2];

    Y = v->s.dest[0];



    dsp->put_pixels_clamped(block[0], Y, ys);

    dsp->put_pixels_clamped(block[1], Y + 8, ys);

    Y += ys * 8;

    dsp->put_pixels_clamped(block[2], Y, ys);

    dsp->put_pixels_clamped(block[3], Y + 8, ys);



    if(!(v->s.flags & CODEC_FLAG_GRAY)) {

        dsp->put_pixels_clamped(block[4], v->s.dest[1], us);

        dsp->put_pixels_clamped(block[5], v->s.dest[2], vs);

    }

}
