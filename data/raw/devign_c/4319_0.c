void qemu_set_dfilter_ranges(const char *filter_spec, Error **errp)

{

    gchar **ranges = g_strsplit(filter_spec, ",", 0);

    int i;



    if (debug_regions) {

        g_array_unref(debug_regions);

        debug_regions = NULL;

    }



    debug_regions = g_array_sized_new(FALSE, FALSE,

                                      sizeof(Range), g_strv_length(ranges));

    for (i = 0; ranges[i]; i++) {

        const char *r = ranges[i];

        const char *range_op, *r2, *e;

        uint64_t r1val, r2val, lob, upb;

        struct Range range;



        range_op = strstr(r, "-");

        r2 = range_op ? range_op + 1 : NULL;

        if (!range_op) {

            range_op = strstr(r, "+");

            r2 = range_op ? range_op + 1 : NULL;

        }

        if (!range_op) {

            range_op = strstr(r, "..");

            r2 = range_op ? range_op + 2 : NULL;

        }

        if (!range_op) {

            error_setg(errp, "Bad range specifier");

            goto out;

        }



        if (qemu_strtoull(r, &e, 0, &r1val)

            || e != range_op) {

            error_setg(errp, "Invalid number to the left of %.*s",

                       (int)(r2 - range_op), range_op);

            goto out;

        }

        if (qemu_strtoull(r2, NULL, 0, &r2val)) {

            error_setg(errp, "Invalid number to the right of %.*s",

                       (int)(r2 - range_op), range_op);

            goto out;

        }



        switch (*range_op) {

        case '+':

            lob = r1val;

            upb = r1val + r2val - 1;

            break;

        case '-':

            upb = r1val;

            lob = r1val - (r2val - 1);

            break;

        case '.':

            lob = r1val;

            upb = r2val;

            break;

        default:

            g_assert_not_reached();

        }

        if (lob > upb || (lob == 0 && upb == UINT64_MAX)) {

            error_setg(errp, "Invalid range");

            goto out;

        }

        range.begin = lob;

        range.end = upb + 1;

        g_array_append_val(debug_regions, range);

    }

out:

    g_strfreev(ranges);

}
