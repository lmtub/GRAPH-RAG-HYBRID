static void toright(unsigned char *dst[3], unsigned char *src[3],

                    int dststride[3], int srcstride[3],

                    int w, int h, struct vf_priv_s* p)

{

        int k;



        for (k = 0; k < 3; k++) {

                unsigned char* fromL = src[k];

                unsigned char* fromR = src[k];

                unsigned char* to = dst[k];

                int src = srcstride[k];

                int dst = dststride[k];

                int ss;

                unsigned int dd;

                int i;



                if (k > 0) {

                        i = h / 4 - p->skipline / 2;

                        ss = src * (h / 4 + p->skipline / 2);

                        dd = w / 4;

                } else {

                        i = h / 2 - p->skipline;

                        ss = src * (h / 2 + p->skipline);

                        dd = w / 2;

                }

                fromR += ss;

                for ( ; i > 0; i--) {

                        int j;

                        unsigned char* t = to;

                        unsigned char* sL = fromL;

                        unsigned char* sR = fromR;



                        if (p->scalew == 1) {

                                for (j = dd; j > 0; j--) {

                                        *t++ = (sL[0] + sL[1]) / 2;

                                        sL+=2;

                                }

                                for (j = dd ; j > 0; j--) {

                                        *t++ = (sR[0] + sR[1]) / 2;

                                        sR+=2;

                                }

                        } else {

                                for (j = dd * 2 ; j > 0; j--)

                                        *t++ = *sL++;

                                for (j = dd * 2 ; j > 0; j--)

                                        *t++ = *sR++;

                        }

                        if (p->scaleh == 1) {

                                fast_memcpy(to + dst, to, dst);

                                to += dst;

                        }

                        to += dst;

                        fromL += src;

                        fromR += src;

                }

                //printf("K %d  %d   %d   %d  %d \n", k, w, h,  src, dst);

        }

}
