void visit_start_alternate(Visitor *v, const char *name,

                           GenericAlternate **obj, size_t size,

                           bool promote_int, Error **errp)

{

    Error *err = NULL;



    assert(obj && size >= sizeof(GenericAlternate));

    assert(v->type != VISITOR_OUTPUT || *obj);

    if (v->start_alternate) {

        v->start_alternate(v, name, obj, size, promote_int, &err);

    }

    if (v->type == VISITOR_INPUT) {

        assert(v->start_alternate && !err != !*obj);

    }

    error_propagate(errp, err);

}
