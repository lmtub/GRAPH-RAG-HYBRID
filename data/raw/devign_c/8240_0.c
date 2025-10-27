void fft_calc_altivec(FFTContext *s, FFTComplex *z)

{

POWERPC_TBL_DECLARE(altivec_fft_num, s->nbits >= 6);

#ifdef ALTIVEC_USE_REFERENCE_C_CODE

    int ln = s->nbits;

    int	j, np, np2;

    int	nblocks, nloops;

    register FFTComplex *p, *q;

    FFTComplex *exptab = s->exptab;

    int l;

    FFTSample tmp_re, tmp_im;

    

POWERPC_TBL_START_COUNT(altivec_fft_num, s->nbits >= 6);

 

    np = 1 << ln;



    /* pass 0 */



    p=&z[0];

    j=(np >> 1);

    do {

        BF(p[0].re, p[0].im, p[1].re, p[1].im, 

           p[0].re, p[0].im, p[1].re, p[1].im);

        p+=2;

    } while (--j != 0);



    /* pass 1 */



    

    p=&z[0];

    j=np >> 2;

    if (s->inverse) {

        do {

            BF(p[0].re, p[0].im, p[2].re, p[2].im, 

               p[0].re, p[0].im, p[2].re, p[2].im);

            BF(p[1].re, p[1].im, p[3].re, p[3].im, 

               p[1].re, p[1].im, -p[3].im, p[3].re);

            p+=4;

        } while (--j != 0);

    } else {

        do {

            BF(p[0].re, p[0].im, p[2].re, p[2].im, 

               p[0].re, p[0].im, p[2].re, p[2].im);

            BF(p[1].re, p[1].im, p[3].re, p[3].im, 

               p[1].re, p[1].im, p[3].im, -p[3].re);

            p+=4;

        } while (--j != 0);

    }

    /* pass 2 .. ln-1 */



    nblocks = np >> 3;

    nloops = 1 << 2;

    np2 = np >> 1;

    do {

        p = z;

        q = z + nloops;

        for (j = 0; j < nblocks; ++j) {

            BF(p->re, p->im, q->re, q->im,

               p->re, p->im, q->re, q->im);

            

            p++;

            q++;

            for(l = nblocks; l < np2; l += nblocks) {

                CMUL(tmp_re, tmp_im, exptab[l].re, exptab[l].im, q->re, q->im);

                BF(p->re, p->im, q->re, q->im,

                   p->re, p->im, tmp_re, tmp_im);

                p++;

                q++;

            }



            p += nloops;

            q += nloops;

        }

        nblocks = nblocks >> 1;

        nloops = nloops << 1;

    } while (nblocks != 0);



POWERPC_TBL_STOP_COUNT(altivec_fft_num, s->nbits >= 6);



#else /* ALTIVEC_USE_REFERENCE_C_CODE */

#ifdef CONFIG_DARWIN

    register const vector float vczero = (const vector float)(0.);

#else

    register const vector float vczero = (const vector float){0.,0.,0.,0.};

#endif

    

    int ln = s->nbits;

    int	j, np, np2;

    int	nblocks, nloops;

    register FFTComplex *p, *q;

    FFTComplex *cptr, *cptr1;

    int k;



POWERPC_TBL_START_COUNT(altivec_fft_num, s->nbits >= 6);



    np = 1 << ln;



    {

        vector float *r, a, b, a1, c1, c2;



        r = (vector float *)&z[0];



        c1 = vcii(p,p,n,n);

        

        if (s->inverse)

            {

                c2 = vcii(p,p,n,p);

            }

        else

            {

                c2 = vcii(p,p,p,n);

            }

        

        j = (np >> 2);

        do {

            a = vec_ld(0, r);

            a1 = vec_ld(sizeof(vector float), r);

            

            b = vec_perm(a,a,vcprmle(1,0,3,2));

            a = vec_madd(a,c1,b);

            /* do the pass 0 butterfly */

            

            b = vec_perm(a1,a1,vcprmle(1,0,3,2));

            b = vec_madd(a1,c1,b);

            /* do the pass 0 butterfly */

            

            /* multiply third by -i */

            b = vec_perm(b,b,vcprmle(2,3,1,0));

            

            /* do the pass 1 butterfly */

            vec_st(vec_madd(b,c2,a), 0, r);

            vec_st(vec_nmsub(b,c2,a), sizeof(vector float), r);

            

            r += 2;

        } while (--j != 0);

    }

    /* pass 2 .. ln-1 */



    nblocks = np >> 3;

    nloops = 1 << 2;

    np2 = np >> 1;



    cptr1 = s->exptab1;

    do {

        p = z;

        q = z + nloops;

        j = nblocks;

        do {

            cptr = cptr1;

            k = nloops >> 1;

            do {

                vector float a,b,c,t1;



                a = vec_ld(0, (float*)p);

                b = vec_ld(0, (float*)q);

                

                /* complex mul */

                c = vec_ld(0, (float*)cptr);

                /*  cre*re cim*re */

                t1 = vec_madd(c, vec_perm(b,b,vcprmle(2,2,0,0)),vczero);

                c = vec_ld(sizeof(vector float), (float*)cptr);

                /*  -cim*im cre*im */

                b = vec_madd(c, vec_perm(b,b,vcprmle(3,3,1,1)),t1);

                

                /* butterfly */

                vec_st(vec_add(a,b), 0, (float*)p);

                vec_st(vec_sub(a,b), 0, (float*)q);

                

                p += 2;

                q += 2;

                cptr += 4;

            } while (--k);

            

            p += nloops;

            q += nloops;

        } while (--j);

        cptr1 += nloops * 2;

        nblocks = nblocks >> 1;

        nloops = nloops << 1;

    } while (nblocks != 0);



POWERPC_TBL_STOP_COUNT(altivec_fft_num, s->nbits >= 6);



#endif /* ALTIVEC_USE_REFERENCE_C_CODE */

}
