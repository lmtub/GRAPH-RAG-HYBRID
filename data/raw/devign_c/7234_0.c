void hmp_info_memdev(Monitor *mon, const QDict *qdict)

{

    Error *err = NULL;

    MemdevList *memdev_list = qmp_query_memdev(&err);

    MemdevList *m = memdev_list;

    StringOutputVisitor *ov;

    char *str;

    int i = 0;





    while (m) {

        ov = string_output_visitor_new(false);

        visit_type_uint16List(string_output_get_visitor(ov), NULL,

                              &m->value->host_nodes, NULL);

        monitor_printf(mon, "memory backend: %d\n", i);

        monitor_printf(mon, "  size:  %" PRId64 "\n", m->value->size);

        monitor_printf(mon, "  merge: %s\n",

                       m->value->merge ? "true" : "false");

        monitor_printf(mon, "  dump: %s\n",

                       m->value->dump ? "true" : "false");

        monitor_printf(mon, "  prealloc: %s\n",

                       m->value->prealloc ? "true" : "false");

        monitor_printf(mon, "  policy: %s\n",

                       HostMemPolicy_lookup[m->value->policy]);

        str = string_output_get_string(ov);

        monitor_printf(mon, "  host nodes: %s\n", str);



        g_free(str);

        string_output_visitor_cleanup(ov);

        m = m->next;

        i++;

    }



    monitor_printf(mon, "\n");



    qapi_free_MemdevList(memdev_list);

}
