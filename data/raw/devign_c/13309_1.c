void visit_start_list(Visitor *v, const char *name, GenericList **list,

                      size_t size, Error **errp)

{

    Error *err = NULL;



    assert(!list || size >= sizeof(GenericList));

    v->start_list(v, name, list, size, &err);

    if (list && v->type == VISITOR_INPUT) {

        assert(!(err && *list));

    }

    error_propagate(errp, err);

}
