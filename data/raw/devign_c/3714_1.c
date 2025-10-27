static void lms_update(WmallDecodeCtx *s, int ich, int ilms, int16_t input, int16_t pred)

{

    int16_t icoef;

    int recent = s->cdlms[ich][ilms].recent;

    int16_t range = 1 << (s->bits_per_sample - 1);

    int bps = s->bits_per_sample > 16 ? 4 : 2; // bytes per sample



    if (input > pred) {

        for (icoef = 0; icoef < s->cdlms[ich][ilms].order; icoef++)

            s->cdlms[ich][ilms].coefs[icoef] +=

                s->cdlms[ich][ilms].lms_updates[icoef + recent];

    } else {

        for (icoef = 0; icoef < s->cdlms[ich][ilms].order; icoef++)

            s->cdlms[ich][ilms].coefs[icoef] -=

                s->cdlms[ich][ilms].lms_updates[icoef];     // XXX: [icoef + recent] ?

    }

    s->cdlms[ich][ilms].recent--;

    s->cdlms[ich][ilms].lms_prevvalues[recent] = av_clip(input, -range, range - 1);



    if (input > pred)

        s->cdlms[ich][ilms].lms_updates[recent] = s->update_speed[ich];

    else if (input < pred)

        s->cdlms[ich][ilms].lms_updates[recent] = -s->update_speed[ich];



    /* XXX: spec says:

    cdlms[iCh][ilms].updates[iRecent + cdlms[iCh][ilms].order >> 4] >>= 2;

    lms_updates[iCh][ilms][iRecent + cdlms[iCh][ilms].order >> 3] >>= 1;



        Questions is - are cdlms[iCh][ilms].updates[] and lms_updates[][][] two

        seperate buffers? Here I've assumed that the two are same which makes

        more sense to me.

    */

    s->cdlms[ich][ilms].lms_updates[recent + s->cdlms[ich][ilms].order >> 4] >>= 2;

    s->cdlms[ich][ilms].lms_updates[recent + s->cdlms[ich][ilms].order >> 3] >>= 1;

    /* XXX: recent + (s->cdlms[ich][ilms].order >> 4) ? */



    if (s->cdlms[ich][ilms].recent == 0) {

        /* XXX: This memcpy()s will probably fail if a fixed 32-bit buffer is used.

                follow kshishkov's suggestion of using a union. */

        memcpy(s->cdlms[ich][ilms].lms_prevvalues + s->cdlms[ich][ilms].order,

               s->cdlms[ich][ilms].lms_prevvalues,

               bps * s->cdlms[ich][ilms].order);

        memcpy(s->cdlms[ich][ilms].lms_updates + s->cdlms[ich][ilms].order,

               s->cdlms[ich][ilms].lms_updates,

               bps * s->cdlms[ich][ilms].order);

        s->cdlms[ich][ilms].recent = s->cdlms[ich][ilms].order;

    }

}
