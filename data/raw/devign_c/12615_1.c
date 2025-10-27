static void use_high_update_speed(WmallDecodeCtx *s, int ich)

{

    int ilms, recent, icoef;

    s->update_speed[ich] = 16;

    for (ilms = s->cdlms_ttl[ich]; ilms >= 0; ilms--) {

        recent = s->cdlms[ich][ilms].recent;

        if (s->bV3RTM) {

            for (icoef = 0; icoef < s->cdlms[ich][ilms].order; icoef++)

                s->cdlms[ich][ilms].lms_updates[icoef + recent] *= 2;

        } else {

            for (icoef = 0; icoef < s->cdlms[ich][ilms].order; icoef++)

                s->cdlms[ich][ilms].lms_updates[icoef] *= 2;

        }

    }

}
