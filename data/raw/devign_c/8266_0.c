opts_start_struct(Visitor *v, void **obj, const char *kind,

                  const char *name, size_t size, Error **errp)

{

    OptsVisitor *ov = DO_UPCAST(OptsVisitor, visitor, v);

    const QemuOpt *opt;



    *obj = g_malloc0(size > 0 ? size : 1);

    if (ov->depth++ > 0) {

        return;

    }



    ov->unprocessed_opts = g_hash_table_new_full(&g_str_hash, &g_str_equal,

                                                 NULL, &destroy_list);

    QTAILQ_FOREACH(opt, &ov->opts_root->head, next) {

        /* ensured by qemu-option.c::opts_do_parse() */

        assert(strcmp(opt->name, "id") != 0);



        opts_visitor_insert(ov->unprocessed_opts, opt);

    }



    if (ov->opts_root->id != NULL) {

        ov->fake_id_opt = g_malloc0(sizeof *ov->fake_id_opt);



        ov->fake_id_opt->name = "id";

        ov->fake_id_opt->str = ov->opts_root->id;

        opts_visitor_insert(ov->unprocessed_opts, ov->fake_id_opt);

    }

}
