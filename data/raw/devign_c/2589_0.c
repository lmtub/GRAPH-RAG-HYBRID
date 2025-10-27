static GenericList *qmp_output_next_list(Visitor *v, GenericList *tail,

                                         size_t size)

{

    return tail->next;

}
