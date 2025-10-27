static void ini_print_object_header(const char *name)

{

    int i;

    PrintElement *el = octx.prefix + octx.level -1;



    if (el->nb_elems)

        avio_printf(probe_out, "\n");



    avio_printf(probe_out, "[");



    for (i = 1; i < octx.level; i++) {

        el = octx.prefix + i;

        avio_printf(probe_out, "%s.", el->name);

        if (el->index >= 0)

            avio_printf(probe_out, "%"PRId64".", el->index);

    }



    avio_printf(probe_out, "%s", name);

    if (el && el->type == ARRAY)

        avio_printf(probe_out, ".%"PRId64"", el->nb_elems);

    avio_printf(probe_out, "]\n");

}
