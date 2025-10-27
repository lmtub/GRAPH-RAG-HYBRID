static av_always_inline int cmp_inline(MpegEncContext *s, const int x, const int y, const int subx, const int suby,

                      const int size, const int h, int ref_index, int src_index,

                      me_cmp_func cmp_func, me_cmp_func chroma_cmp_func, int qpel, int chroma){

    MotionEstContext * const c= &s->me;

    const int stride= c->stride;

    const int uvstride= c->uvstride;

    const int dxy= subx + (suby<<(1+qpel)); //FIXME log2_subpel?

    const int hx= subx + (x<<(1+qpel));

    const int hy= suby + (y<<(1+qpel));

    uint8_t * const * const ref= c->ref[ref_index];

    uint8_t * const * const src= c->src[src_index];

    int d;

    //FIXME check chroma 4mv, (no crashes ...)

        int uvdxy;              /* no, it might not be used uninitialized */

        if(dxy){

            if(qpel){

                c->qpel_put[size][dxy](c->temp, ref[0] + x + y*stride, stride); //FIXME prototype (add h)

                if(chroma){

                    int cx= hx/2;

                    int cy= hy/2;

                    cx= (cx>>1)|(cx&1);

                    cy= (cy>>1)|(cy&1);

                    uvdxy= (cx&1) + 2*(cy&1);

                    //FIXME x/y wrong, but mpeg4 qpel is sick anyway, we should drop as much of it as possible in favor for h264

                }

            }else{

                c->hpel_put[size][dxy](c->temp, ref[0] + x + y*stride, stride, h);

                if(chroma)

                    uvdxy= dxy | (x&1) | (2*(y&1));

            }

            d = cmp_func(s, c->temp, src[0], stride, h);

        }else{

            d = cmp_func(s, src[0], ref[0] + x + y*stride, stride, h);

            if(chroma)

                uvdxy= (x&1) + 2*(y&1);

        }

        if(chroma){

            uint8_t * const uvtemp= c->temp + 16*stride;

            c->hpel_put[size+1][uvdxy](uvtemp  , ref[1] + (x>>1) + (y>>1)*uvstride, uvstride, h>>1);

            c->hpel_put[size+1][uvdxy](uvtemp+8, ref[2] + (x>>1) + (y>>1)*uvstride, uvstride, h>>1);

            d += chroma_cmp_func(s, uvtemp  , src[1], uvstride, h>>1);

            d += chroma_cmp_func(s, uvtemp+8, src[2], uvstride, h>>1);

        }

    return d;

}
