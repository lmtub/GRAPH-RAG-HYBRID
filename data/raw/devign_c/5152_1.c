void visit_optional(Visitor *v, bool *present, const char *name,

                    Error **errp)

{

    if (!error_is_set(errp) && v->optional) {

        v->optional(v, present, name, errp);

    }

}
