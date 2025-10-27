static void string_output_append(StringOutputVisitor *sov, int64_t a)

{

    Range *r = g_malloc0(sizeof(*r));

    r->begin = a;

    r->end = a + 1;

    sov->ranges = g_list_insert_sorted_merged(sov->ranges, r, range_compare);

}
