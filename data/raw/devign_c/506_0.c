GenericList *visit_next_list(Visitor *v, GenericList **list, size_t size)

{

    assert(list && size >= sizeof(GenericList));

    return v->next_list(v, list, size);

}
