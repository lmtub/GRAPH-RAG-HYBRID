start_list(Visitor *v, const char *name, Error **errp)

{

    StringInputVisitor *siv = to_siv(v);



    if (parse_str(siv, name, errp) < 0) {

        return;

    }



    siv->cur_range = g_list_first(siv->ranges);

    if (siv->cur_range) {

        Range *r = siv->cur_range->data;

        if (r) {

            siv->cur = r->begin;

        }

    }

}
