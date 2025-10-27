static inline int16_t g726_iterate(G726Context* c, int16_t I)

{

    int dq, re_signal, pk0, fa1, i, tr, ylint, ylfrac, thr2, al, dq0;

    Float11 f;

    

    dq = inverse_quant(c, I);

    if (I >> (c->tbls->bits - 1))  /* get the sign */

        dq = -dq;

    re_signal = c->se + dq;



    /* Transition detect */

    ylint = (c->yl >> 15);

    ylfrac = (c->yl >> 10) & 0x1f;

    thr2 = (ylint > 9) ? 0x1f << 10 : (0x20 + ylfrac) << ylint;

    if (c->td == 1 && abs(dq) > ((thr2+(thr2>>1))>>1))

        tr = 1;

    else

        tr = 0;

    

    /* Update second order predictor coefficient A2 and A1 */

    pk0 = (c->sez + dq) ? sgn(c->sez + dq) : 0;

    dq0 = dq ? sgn(dq) : 0;

    if (tr) {

        c->a[0] = 0;

	c->a[1] = 0;

        for (i=0; i<6; i++)

	   c->b[i] = 0;

    } else {

	/* This is a bit crazy, but it really is +255 not +256 */

	fa1 = clamp((-c->a[0]*c->pk[0]*pk0)>>5, -256, 255);

	

	c->a[1] += 128*pk0*c->pk[1] + fa1 - (c->a[1]>>7);

	c->a[1] = clamp(c->a[1], -12288, 12288);

        c->a[0] += 64*3*pk0*c->pk[0] - (c->a[0] >> 8);

	c->a[0] = clamp(c->a[0], -(15360 - c->a[1]), 15360 - c->a[1]);



        for (i=0; i<6; i++)

	     c->b[i] += 128*dq0*sgn(-c->dq[i].sign) - (c->b[i]>>8);

    }



    /* Update Dq and Sr and Pk */

    c->pk[1] = c->pk[0];

    c->pk[0] = pk0 ? pk0 : 1;

    c->sr[1] = c->sr[0];

    i2f(re_signal, &c->sr[0]);

    for (i=5; i>0; i--)

       c->dq[i] = c->dq[i-1];

    i2f(dq, &c->dq[0]);

    c->dq[0].sign = I >> (c->tbls->bits - 1); /* Isn't it crazy ?!?! */

    

    /* Update tone detect [I'm not sure 'tr == 0' is really needed] */

    c->td = (tr == 0 && c->a[1] < -11776); 

       

    /* Update Ap */

    c->dms += ((c->tbls->F[I]<<9) - c->dms) >> 5;

    c->dml += ((c->tbls->F[I]<<11) - c->dml) >> 7;

    if (tr) 

       c->ap = 256;

    else if (c->y > 1535 && !c->td && (abs((c->dms << 2) - c->dml) < (c->dml >> 3)))

       c->ap += (-c->ap) >> 4;

    else

       c->ap += (0x200 - c->ap) >> 4; 



    /* Update Yu and Yl */

    c->yu = clamp(c->y + (((c->tbls->W[I] << 5) - c->y) >> 5), 544, 5120);

    c->yl += c->yu + ((-c->yl)>>6);

 

    /* Next iteration for Y */

    al = (c->ap >= 256) ? 1<<6 : c->ap >> 2;

    c->y = (c->yl + (c->yu - (c->yl>>6))*al) >> 6;

	

    /* Next iteration for SE and SEZ */

    c->se = 0;

    for (i=0; i<6; i++)

       c->se += mult(i2f(c->b[i] >> 2, &f), &c->dq[i]);

    c->sez = c->se >> 1;

    for (i=0; i<2; i++)

       c->se += mult(i2f(c->a[i] >> 2, &f), &c->sr[i]);

    c->se >>= 1;



    return clamp(re_signal << 2, -0xffff, 0xffff);

}
