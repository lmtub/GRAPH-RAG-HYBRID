void visit_type_number(Visitor *v, double *obj, const char *name, Error **errp)

{

    if (!error_is_set(errp)) {

        v->type_number(v, obj, name, errp);

    }

}
