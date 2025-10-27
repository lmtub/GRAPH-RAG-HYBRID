void visit_end_struct(Visitor *v, Error **errp)

{

    assert(!error_is_set(errp));

    v->end_struct(v, errp);

}
