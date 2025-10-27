void visit_type_size(Visitor *v, uint64_t *obj, const char *name, Error **errp)

{

    int64_t value;



    if (v->type_size) {

        v->type_size(v, obj, name, errp);

    } else if (v->type_uint64) {

        v->type_uint64(v, obj, name, errp);

    } else {

        value = *obj;

        v->type_int64(v, &value, name, errp);

        *obj = value;

    }

}
