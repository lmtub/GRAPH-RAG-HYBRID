static void rac_normalise(RangeCoder *c)

{

    for (;;) {

        c->range <<= 8;

        c->low   <<= 8;

        if (c->src < c->src_end) {

            c->low |= *c->src++;

        } else if (!c->low) {

            c->got_error = 1;

            return;

        }

        if (c->range >= RAC_BOTTOM)

            return;

    }

}
