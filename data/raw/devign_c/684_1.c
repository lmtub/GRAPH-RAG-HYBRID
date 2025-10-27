void visit_type_str(Visitor *v, char **obj, const char *name, Error **errp)

{

    if (!error_is_set(errp)) {

        v->type_str(v, obj, name, errp);

    }

}
