void qdist_bin__internal(struct qdist *to, const struct qdist *from, size_t n)

{

    double xmin, xmax;

    double step;

    size_t i, j;



    qdist_init(to);



    if (from->n == 0) {

        return;

    }

    if (n == 0 || from->n == 1) {

        n = from->n;

    }



    /* set equally-sized bins between @from's left and right */

    xmin = qdist_xmin(from);

    xmax = qdist_xmax(from);

    step = (xmax - xmin) / n;



    if (n == from->n) {

        /* if @from's entries are equally spaced, no need to re-bin */

        for (i = 0; i < from->n; i++) {

            if (from->entries[i].x != xmin + i * step) {

                goto rebin;

            }

        }

        /* they're equally spaced, so copy the dist and bail out */

        to->entries = g_new(struct qdist_entry, from->n);

        to->n = from->n;

        memcpy(to->entries, from->entries, sizeof(*to->entries) * to->n);

        return;

    }



 rebin:

    j = 0;

    for (i = 0; i < n; i++) {

        double x;

        double left, right;



        left = xmin + i * step;

        right = xmin + (i + 1) * step;



        /* Add x, even if it might not get any counts later */

        x = left;

        qdist_add(to, x, 0);



        /*

         * To avoid double-counting we capture [left, right) ranges, except for

         * the righmost bin, which captures a [left, right] range.

         */

        while (j < from->n && (from->entries[j].x < right || i == n - 1)) {

            struct qdist_entry *o = &from->entries[j];



            qdist_add(to, x, o->count);

            j++;

        }

    }

}
