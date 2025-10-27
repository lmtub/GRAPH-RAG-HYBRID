void qemu_set_dfilter_ranges(const char *filter_spec)
{
    gchar **ranges = g_strsplit(filter_spec, ",", 0);
    if (ranges) {
        gchar **next = ranges;
        gchar *r = *next++;
        debug_regions = g_array_sized_new(FALSE, FALSE,
                                          sizeof(Range), g_strv_length(ranges));
        while (r) {
            char *range_op = strstr(r, "-");
            char *r2 = range_op ? range_op + 1 : NULL;
            if (!range_op) {
                range_op = strstr(r, "+");
                r2 = range_op ? range_op + 1 : NULL;
            if (!range_op) {
                range_op = strstr(r, "..");
                r2 = range_op ? range_op + 2 : NULL;
            if (range_op) {
                const char *e = NULL;
                uint64_t r1val, r2val;
                if ((qemu_strtoull(r, &e, 0, &r1val) == 0) &&
                    (qemu_strtoull(r2, NULL, 0, &r2val) == 0) &&
                    r2val > 0) {
                    struct Range range;
                    g_assert(e == range_op);
                    switch (*range_op) {
                    case '+':
                    {
                        range.begin = r1val;
                        range.end = r1val + (r2val - 1);
                        break;
                    case '-':
                    {
                        range.end = r1val;
                        range.begin = r1val - (r2val - 1);
                        break;
                    case '.':
                        range.begin = r1val;
                        range.end = r2val;
                        break;
                    default:
                        g_assert_not_reached();
                    g_array_append_val(debug_regions, range);
                } else {
                    g_error("Failed to parse range in: %s", r);
            } else {
                g_error("Bad range specifier in: %s", r);
            r = *next++;
        g_strfreev(ranges);