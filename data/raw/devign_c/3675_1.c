void visit_start_struct(Visitor *v, void **obj, const char *kind,

                        const char *name, size_t size, Error **errp)

{

    if (!error_is_set(errp)) {

        v->start_struct(v, obj, kind, name, size, errp);

    }

}
