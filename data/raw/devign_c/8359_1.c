void visit_type_enum(Visitor *v, const char *name, int *obj,

                     const char *const strings[], Error **errp)

{

    assert(obj && strings);

    if (v->type == VISITOR_INPUT) {

        input_type_enum(v, name, obj, strings, errp);

    } else if (v->type == VISITOR_OUTPUT) {

        output_type_enum(v, name, obj, strings, errp);

    }

}
