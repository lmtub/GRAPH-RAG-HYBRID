GenericList *visit_next_list(Visitor *v, GenericList **list, Error **errp)

{

    if (!error_is_set(errp)) {

        return v->next_list(v, list, errp);

    }



    return 0;

}
