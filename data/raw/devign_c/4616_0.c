static void string_output_append_range(StringOutputVisitor *sov,

                                       int64_t s, int64_t e)

{

    Range *r = g_malloc0(sizeof(*r));

    r->begin = s;

    r->end = e + 1;

    sov->ranges = range_list_insert(sov->ranges, r);

}
