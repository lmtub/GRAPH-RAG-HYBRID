opts_next_list(Visitor *v, GenericList **list, Error **errp)

{

    OptsVisitor *ov = DO_UPCAST(OptsVisitor, visitor, v);

    GenericList **link;



    if (ov->repeated_opts_first) {

        ov->repeated_opts_first = false;

        link = list;

    } else {

        const QemuOpt *opt;



        opt = g_queue_pop_head(ov->repeated_opts);

        if (g_queue_is_empty(ov->repeated_opts)) {

            g_hash_table_remove(ov->unprocessed_opts, opt->name);

            return NULL;

        }

        link = &(*list)->next;

    }



    *link = g_malloc0(sizeof **link);

    return *link;

}
