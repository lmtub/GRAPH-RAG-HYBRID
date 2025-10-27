static void string_output_append(StringOutputVisitor *sov, int64_t a)

{

    Range *r = g_malloc0(sizeof(*r));

    r->begin = a;

    r->end = a + 1;

    sov->ranges = range_list_insert(sov->ranges, r);

}
