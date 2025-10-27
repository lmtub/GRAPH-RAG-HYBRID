void visit_start_list(Visitor *v, const char *name, Error **errp)

{

    if (!error_is_set(errp)) {

        v->start_list(v, name, errp);

    }

}
